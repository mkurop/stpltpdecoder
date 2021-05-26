# -*- coding: utf-8
 
__author__ = 'Marcin Kuropatwiński'

"""Module for the computation of the Short Term Predictor and the Long Term Predictor parameters. \
Provided classes for encoder and decoder work for any specified sampling frequency.

Created 31.03.2014 Code modernized 22.05.2021"""

import os
import shutil
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import deconvolve, lfilter, lfiltic

from spectrum import *
import decoder_old.decoder


def lpanafil(s : np.ndarray, a : np.ndarray, hlp : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the short term residual given Short Term Predictor (STP) parameters and an initial state.

    :param s: signal frame
    :type s: np.ndarray
    :param a: STP parameters [1, a1 , ..., a_p]
    :type a: np.ndarray
    :param hlp: history of the analysis filter
    :type hlp: np.ndarray
    :return: 
        * res - STP residual
        * hlp - history for the next signal frame
    :rtype: Tuple[np.ndarray, np.ndarray]
    """

    s = np.concatenate([hlp,s])
    p = len(a)-1
    res = np.convolve(s,a,mode='full')
    res = res[p:-p]
    hlp = s[-p:]
    return res, hlp


class Config:
    """Class holding the parameters of the STP/LTP encoder/decoder. Configurable using single \
    variable representing the sampling rate of the processed signal.

    :param sampling_rate: sampling rate in Hz of the signal processed
    :type sampling_rate: int
    :raises ValueError: raised if sampling_rate is outside the set {8000, 16000}
    :return: the constructor does not return
    """

    def __init__(self, sampling_rate: int):
        """Constructor method"""

        if not sampling_rate in set([8000, 16000]):

            raise ValueError("Sampling rate have to be either 8000Hz or 16000Hz.")

        # memorize the sampling_rate
        self.sampling_rate = sampling_rate

        # order of the STP filter

        self.p = 10 if sampling_rate == 8000 else 16 

        # length of the signal frame in samples

        self.frame = 160 if sampling_rate == 8000 else 320
        
        # number of subframes in a frame

        self.subframes = 4

        # length of the signal subframe in samples

        self.subframe = int(self.frame / self.subframes)

        # minimal and maximal pitch lag
        self.pitch_lag_max = int(120 * sampling_rate/8000)
        self.pitch_lag_min = int(20 * sampling_rate/8000)

        # parameters for the poles bandwidth widening procedure
        self.g1 = 0.994
        self.gk1 = np.cumprod(np.ones((self.p,)) * self.g1)

class OutputParameters:
    """Empty class for holding the output parameters from the encoder."""
    pass

class StpLtpEncoder:
    """Class performing the STP/LTP analysis of the input frame given the state of the encoder

    :param sampling_rate: sampling rate in Hz of the signal processed
    :type sampling_rate: int
    """

    class EncoderState:
        """Class holding the state of encoder, which changes from frame to frame

        :param sampling_rate: sampling rate in Hz of the signal processed
        :type sampling_rate: int
        """

        def __init__(self, sampling_rate: int):
            """Constructor method"""

            self.cfg = Config(sampling_rate)

            self.reset_state()

        def reset_state(self):

            self.lsf = np.linspace(0.2, np.pi - 0.1, self.cfg.p) # LSF parameters from the previous frame

            self.hlp = np.zeros((self.cfg.p,)) # initial conditions for the STP analysis 

            self.short_term_res = np.random.randn(self.cfg.pitch_lag_max,)*1e-6 # initial conditions for the LTP analysis

        def set_lsf(self, lsf : np.ndarray):

            self.lsf = lsf

        def set_hlp(self, hlp: np.ndarray):

            self.hlp = hlp

        def set_short_term_residual_initial_conditions(self, short_term_res : np.ndarray):

            self.short_term_res = short_term_res

    def __init__(self, sampling_rate: int = 16000):
        """Constructor method"""

        # instantiate object of type Config
        
        self.cfg = Config(sampling_rate)

        # instantiate encoder state object

        self.state = StpLtpEncoder.EncoderState(sampling_rate)

    def frame(self, signal_frame : np.ndarray):
        """ Analyzes the input signal frame and returns a structure of output parameters, updates the state object.

        :param signal_frame: samples in the signal frame
        :type signal_frame: np.ndarray
        :return: structure with fields
            * lsf - the line spectral frequency parameters in subframes, matrix with dimension p x subframes  \
            where subframes is the number of subframes per frame and p is the STP order
            * a - the STP polynomial coefficients in subframes, matrix with dimension (p+1) x subframes 
            * ltp_lags - the LTP predictor lags
            * ltp_taps - the LTP predictor taps
            * ltp_variances - variances of the excitation
            * current_frame_long_term_res - the LTP residual, the excitation signal
        """

        if signal_frame.size != self.cfg.frame:
            raise ValueError("The signal_frame input variable have to be an nd.array with {self.cfg.frame} elements.")

        # STP analysis ---------------------------

        # compute autocorrelation of the input frame
        x, lags = xcorr(signal_frame, maxlags = self.cfg.p)

        # compute the STP polynomial
        a, stp_excitation_variance, reflection_coefficients = levinson(x[self.cfg.p:]) 

        a = np.concatenate(([1.],a))

        # bandwidth extension
        a[1:] *= self.cfg.gk1

        # conversion of the STP polynomial into Line Spectral Frequencies
        lsf = poly2lsf(a)

        # compute interpolated LSFs
        lsf_from_previous_frame = np.tile(self.state.lsf,(self.cfg.subframes,1)).T
        lsf_from_current_frame = np.tile(lsf,(self.cfg.subframes,1)).T

        lsf_interpolated = lsf_from_previous_frame * np.linspace(1-1./self.cfg.subframes,0,self.cfg.subframes) + \
                           lsf_from_current_frame * np.linspace(1./self.cfg.subframes,1,self.cfg.subframes)  

        # convert interpolated LSFs back to STP polynomials
        a_interpolated = np.asarray([lsf2poly(lsf_interpolated[:,i].ravel()) for i in range(self.cfg.subframes)])

        # compute short term predictor residual
        current_frame_short_term_res = np.zeros((self.cfg.frame,))
        for i in range(self.cfg.subframes):

            current_frame_short_term_res[i*self.cfg.subframe:(i+1)*self.cfg.subframe], self.state.hlp = \
                lpanafil(signal_frame[i*self.cfg.subframe:(i+1)*self.cfg.subframe], a_interpolated[i,:].ravel(), self.state.hlp)

        # LTP analysis -------------------------

        short_term_residual = np.concatenate((self.state.short_term_res,current_frame_short_term_res))

        # space for the long term residual
        current_frame_long_term_res = np.zeros_like(current_frame_short_term_res)

        # space for pitches in subframes (LTP lags)
        ltp_lags = np.zeros((self.cfg.subframes,))

        # space for LTP taps
        ltp_taps = np.zeros_like(ltp_lags)

        # space for LTP variances
        ltp_variances = np.zeros_like(ltp_lags)
        
        for i in range(self.cfg.subframes):

            current_subframe = current_frame_short_term_res[i*self.cfg.subframe:(i+1)*self.cfg.subframe]

            past_short_term_residual = short_term_residual[i*self.cfg.subframe: (i+1)*self.cfg.subframe + self.cfg.pitch_lag_max - self.cfg.pitch_lag_min]

            # long term correlations
            aux_matrix1 = np.expand_dims(np.arange(self.cfg.subframe),0) + np.expand_dims(np.arange(self.cfg.pitch_lag_max-self.cfg.pitch_lag_min+1),0).T

            #  print(aux_matrix1)

            aux_matrix2 = past_short_term_residual[aux_matrix1] 

            #  print(aux_matrix2)
            #  input('PE')

            long_term_correlations = np.sum(aux_matrix2 * current_subframe,axis=1)

            denominators = np.sum(aux_matrix2 * aux_matrix2,axis=1)

            merit = long_term_correlations/denominators

            aux_matrix3 = merit * long_term_correlations

            aux1 = np.argmax(aux_matrix3)

            print(aux_matrix3[aux1])

            # fill the pitch for the subframe
            ltp_lags[i] = self.cfg.pitch_lag_max - aux1
            
            # fill the LTP filter tap for the subframe
            ltp_taps[i] = -merit[aux1]
            
            # fill the variance of the long term residual for the subframe
            ltp_variances[i] = (np.sum(current_subframe**2) - aux_matrix3[aux1])/self.cfg.subframe

            current_frame_long_term_res[i*self.cfg.subframe:(i+1)*self.cfg.subframe] = (current_subframe + ltp_taps[i]*aux_matrix2[aux1,:].ravel())/np.sqrt(ltp_variances[i])

        #  fig, ax = plt.subplots(figsize=(6,6))
        #  ax.plot(signal_frame, label="signal")
        #  ax.plot(current_frame_short_term_res, label="STP residual")
        #  ax.plot(current_frame_long_term_res, label="LTP residual")
        #  plt.ylim([signal_frame.min(),signal_frame.max()])
        #  ax.legend()
        #  plt.show()

        # update state for next frame
        self.state.lsf = lsf
        self.state.short_term_res = current_frame_short_term_res[-self.cfg.pitch_lag_max:]
        
        # fill output parameters structure
        output_parameters = OutputParameters()
        output_parameters.lsf = lsf_interpolated
        output_parameters.a = a_interpolated
        
        output_parameters.ltp_lags = np.uint32(ltp_lags) 
        output_parameters.ltp_taps = ltp_taps
        output_parameters.ltp_variances = ltp_variances
        output_parameters.current_frame_long_term_res = current_frame_long_term_res

        return output_parameters, short_term_residual[self.cfg.pitch_lag_max:]


class StpLtpDecoder:

    class DecoderState:
        """Class holding the state of decoder, which changes from frame to frame

        :param sampling_rate: sampling rate in Hz of the signal processed
        :type sampling_rate: int
        """

        def __init__(self, sampling_rate : int):
            """Constructor method"""

            self.cfg = Config(sampling_rate)

            self.reset_state()

        def reset_state(self):

            self.lsf = np.linspace(0.2, np.pi - 0.1, self.cfg.p) 

            a = lsf2poly(self.lsf)

            self.hlp = lfiltic([1.0], a, np.zeros((self.cfg.p,))) 

            self.short_term_res = np.random.randn(self.cfg.pitch_lag_max,)*1e-6

        def set_lsf(self, lsf : np.ndarray):

            self.lsf = lsf

            self.a = lsf2poly(self.lsf)

        def set_hlp(self, hlp: np.ndarray, a: np.ndarray):

            self.hlp = lfiltic([1.0], a, hlp[::-1])

        def set_short_term_residual_initial_conditions(self, short_term_res : np.ndarray):

            self.short_term_res = short_term_res

    def __init__(self, sampling_rate : int):

        self.cfg = Config(sampling_rate) 

        self.state = StpLtpDecoder.DecoderState(sampling_rate)

    def _short_term_residual_synthesis(self, parameters):

        short_term_residual = np.zeros((self.cfg.frame,))

        for i in range(self.cfg.subframes):

            # LTP synthesis -------------------------------

            long_term_residual_subframe = parameters.current_frame_long_term_res[i*self.cfg.subframe:(i+1)*self.cfg.subframe]

            short_term_residual[i*self.cfg.subframe:(i+1)*self.cfg.subframe] = \
                self._ltp_single_subframe_synthesis(parameters, long_term_residual_subframe, i)
        
        return short_term_residual


    def _synthesize_chunk(self, short_term_residual, parameters, i, up_to_sample):

        remainder = self.cfg.pitch_lag_max + self.cfg.subframe - up_to_sample

        beg1 = up_to_sample - parameters.ltp_lags[i]

        end1 = np.amin((up_to_sample, beg1 + remainder))

        len1 = end1 - beg1

        beg2 = up_to_sample 

        end2 = np.amin((beg2 + len1, self.cfg.pitch_lag_max + self.cfg.subframe))

        #  print(beg1, end1)
        #  print(len1)
        #  print(beg2, end2)
        #  print(end2-beg2)
        #  print(beg2 - beg1)
        #  print(parameters.ltp_lags[i])
        #  print(f'short term state + subframe length {self.cfg.pitch_lag_max+self.cfg.subframe}')
        #  print(f'new up_to_sample {end2}')
        #  print(f'up_to_sample = {up_to_sample}')
        #  input('PE chunk')
        #
        #  print(short_term_residual[beg2:end2].size, short_term_residual[beg1:end1].size)
        #  input('size')
        short_term_residual[beg2:end2] -= parameters.ltp_taps[i] * short_term_residual[beg1:end1]



        #  print(end2)
        #  print(self.cfg.pitch_lag_max+self.cfg.subframe)

        if end2 == self.cfg.pitch_lag_max + self.cfg.subframe:


            # update state
            self.state.short_term_res = short_term_residual[self.cfg.subframe:]

            self.short_term_residual_subframe = short_term_residual[self.cfg.pitch_lag_max:]

            return True, end2
        else:

            return False, end2 
            
    def _ltp_single_subframe_synthesis(self, parameters, long_term_residual_subframe: np.ndarray, i : int):

        short_term_residual_subframe = np.sqrt(parameters.ltp_variances[i]) * long_term_residual_subframe 

        # extend current frame short_term_residual with the past samples of the STP residual
        short_term_residual = np.concatenate((self.state.short_term_res,short_term_residual_subframe))

        flag, up_to_sample = self._synthesize_chunk(short_term_residual, parameters, i, self.cfg.pitch_lag_max)

        while not flag:

            flag, up_to_sample =self._synthesize_chunk(short_term_residual, parameters, i, up_to_sample) 

            #  print(up_to_sample)
            #  input('PE')

        return self.short_term_residual_subframe

    def _signal_synthesis(self, parameters, short_term_residual):

        output_signal_frame = np.zeros((self.cfg.frame,))

        # recreate the output frame and update initial conditions for the STP synthesis filter

        for i in range(self.cfg.subframes):

            [output_signal_frame[i*self.cfg.subframe:(i+1)*self.cfg.subframe], self.state.hlp] = \
                lfilter([1.0],parameters.a[i,:].ravel(),\
                        short_term_residual[i*self.cfg.subframe:(i+1)*self.cfg.subframe],zi = self.state.hlp)

        return output_signal_frame

    def frame(self, parameters) -> np.ndarray:
        """STP/LTP synthesis procedure for one frame

        :param parameters: an input parameters structure with the fields
            * lsf - the line spectral frequency parameters in subframes, matrix with dimension p x subframes  \
            where subframes is the number of subframes per frame and p is the STP order
            * a - the STP polynomial coefficients in subframes, matrix with dimension (p+1) x subframes 
            * ltp_lags - the LTP predictor lags
            * ltp_taps - the LTP predictor taps
            * ltp_variances - variances of the excitation
            * current_frame_long_term_res - the LTP residual, the excitation signal

        :return: decoder output signal frame
        :rtype: np.ndarray 
        """

        # LTP synthesis

        short_term_residual = self._short_term_residual_synthesis(parameters)

        # STP synthesis ----------------------------------
        
        output_signal_frame = self._signal_synthesis(parameters, short_term_residual)

        #  fig, ax = plt.subplots(figsize=(6,6))
        #  ax.plot(output_signal_frame, label='signal frame') #'STP synthesised residual'
        #  plt.ylim([-0.3,0.3])
        #  plt.show()
        return output_signal_frame, short_term_residual

def wav_files_in_directory(dir : str) -> List[str]:
    """ Finds all files with the .wav extention in subdirectories of the dir folder.

    :param dir: the folder to be searched over
    :type dir: str

    :return: the list of files with .wav extension found
    :rtype: List[str]

    :raises ValueError: error raised if the provided directory does not exist
    """
    
    if not os.path.exists(dir):
        raise ValueError(f"The directory {dir} does not exists.")

    files_list = []

    for roots,dirs,files in os.walk(dir):

        for file_ in files:

            if file_.endswith("WAV") or file_.endswith("wav"):

                files_list.append(os.path.join(roots,file_))

    return files_list

def read_wav_and_normalize(file_ : str) -> Tuple[np.ndarray,int]:

    if not os.path.exists(file_):
        raise ValueError(f"The file {file_} does not exist.")

    sr, s = wavfile.read(file_)

    s = s/2.**15
    

    np.random.seed(1000) # causes the dither is same on each run

    s += np.random.randn(*s.shape)*1.e-6 # add dither to improve numerical behaviour

    return s, int(sr)

def write_wav(signal : np.ndarray, sampling_rate : int, file_name : str):

    try:
        os.makedirs(os.path.dirname(file_name))
    except OSError:
        print(f'Can not create the directory {os.path.dirname(file_name)}')
    else:
        print(f'Path {os.path.dirname(file_name)} succesfully created.')

    wavfile.write(file_name, sampling_rate, np.int16(signal*2**15))

def clear_output_directory():

    if os.path.exists('../data/output/'): 
        if len(os.listdir('../data/output/')) > 0:
            shutil.rmtree('../data/output') 
    else: 
        try:
            os.makedirs('../data/output/')
        except OSError:
            print("Creation of ../data/output/ failed")
        else:
            print("Successfully created the directory ../data/output/")

def frames_generator(s : np.ndarray, frame_length):

    start = 0

    while start + frame_length < s.size:

        yield s[start:start+frame_length]

        start += frame_length



if __name__ == "__main__":

    files = wav_files_in_directory('../data/input/')

    clear_output_directory()

    sample_rate = 16000

    #  for file_ in files[:1]:
#
        #  speech_signal, sample_rate = read_wav_and_normalize(file_)
#
        #  encode = StpLtpEncoder(sample_rate)
#
        #  decode = StpLtpDecoder(sample_rate)
#
        #  output_signal = np.zeros_like(speech_signal)
#
        #  start = 0
#
        #  for signal_frame in frames_generator(speech_signal, encode.cfg.frame):
    
    # load data

    sig_prev = np.loadtxt('sig_previous.txt')

    sig_current = np.loadtxt('sig_current.txt')

    str_previous = np.loadtxt('str_previous.txt')

    lsf_previous = np.loadtxt('lsf_previous.txt')

    sig = np.concatenate((sig_prev, sig_current))

    signal_frame = sig_current

    # setup encoder

    encode = StpLtpEncoder(sample_rate)

    encode.state.set_lsf(lsf_previous)
#
    encode.state.set_hlp(sig_prev[-encode.cfg.p:])

    encode.state.set_short_term_residual_initial_conditions(str_previous[-encode.cfg.pitch_lag_max:])

    # setup decoder

    decode = StpLtpDecoder(sample_rate)

    decode.state.set_lsf(lsf_previous)

    decode.state.set_short_term_residual_initial_conditions(str_previous[-encode.cfg.pitch_lag_max:])

    # setup old encoder

    CodStat = decoder_old.decoder.InitEncStatFromData(16, 320, lsf_previous, lsf_previous, sig_prev[-encode.cfg.p:], str_previous)

    fig, axs = plt.subplots(1,2,figsize = (18,6))

    axs[0].plot(sig)

    axs[1].plot(str_previous)

    plt.show()


    # encode

    parameters, short_term_residual_encoded = encode.frame(signal_frame)

    print(parameters.ltp_lags)
    print(parameters.ltp_taps)
    print(f"LtpVar new {parameters.ltp_variances}")
    print(parameters.lsf[:,3])
    print(f"STP {parameters.a}")

    # encode with the old encoder

    CodStat, Par, LongTermResidual = decoder_old.decoder.Encoder(signal_frame, CodStat, 16, 320, 4, 4, 16000)

    print(Par.LtpLag)
    print(Par.LtpTap)
    print(f"LtpVar {Par.LtpVar}")

    fig, axs = plt.subplots(figsize=(6,6))
    axs.plot(parameters.current_frame_long_term_res, label="new encoder")
    #  axs.plot(short_term_residual_encoded, label="new encoder")
    axs.plot(LongTermResidual, label="old encoder")
    #  axs.plot(CodStat.ShortTermRes, label="old encoder")
    plt.legend()
    plt.show()

    #  print(parameters.ltp_lags)

    #  output_signal[start:start+encode.cfg.frame], short_term_residual_decoded = decode.frame(parameters)
    print(f"shape a : {parameters.a.shape}")
    input('PE')

    decode.state.set_hlp(sig_prev[-encode.cfg.p:], parameters.a[0,:])
    output_frame, short_term_residual_decoded = decode.frame(parameters)

    fig, axs = plt.subplots(figsize=(6,6))
    axs.plot(short_term_residual_decoded,label='decoded')
    axs.plot(short_term_residual_encoded,label='encoded')
    plt.legend()
    plt.show()
    print(output_frame)
    fig, axs = plt.subplots(figsize=(6,6))
    axs.plot(output_frame, label='decoded')
    axs.plot(signal_frame, label='input')
    plt.legend()
    plt.show()

#              if start == 7040 - encode.cfg.frame:
#
#                  np.savetxt('str_previous.txt', short_term_residual_encoded)
#
#                  np.savetxt('sig_previous.txt', signal_frame)
#
#                  np.savetxt('lsf_previous.txt', parameters.lsf[:,3])
#
#              if start == 7040:
#
#                  np.savetxt('sig_current.txt', signal_frame)
#
#              print(f"Start of the frame {start}, end {start+encode.cfg.frame}")
#
#              fig, ax = plt.subplots(figsize = (6,6))
#
#              ax.plot(short_term_residual_encoded, label='signal')
#
#              ax.plot(signal_frame, label='reconstructed signal')
#  #
#              ax.legend()
#
#              plt.show()
#
#              start += encode.cfg.frame

    #  print(file_.replace('input','output'))
    #
    #  fig, ax = plt.subplots(figsize = (6,6))
    #  #ax.plot(s)
    #  ax.plot(output_signal)
    #  plt.show()
    #  write_wav(output_signal, encode.cfg.sampling_rate, file_.replace('input','output'))
            




        
    #  sr, s = wavfile.read("VLRecording10591_05.04.13_25.168.wav")
    #  s = s/2.**15
    #  s_out = np.zeros_like(s)
    #  CodStat = InitEncStat(16, 512)
    #  DecStat = InitDecStat(16, 512)
    #  start = 0
    #  Frame = 512
    #  p = 16
    #  while start + Frame < len(s):
    #      fs = s[start:start+Frame]
    #      CodStat, Par, LongTermRes = Encoder(fs, CodStat, p, Frame, 4, 4, 16000)
    #      fs_out, DecStat = Decoder(LongTermRes,Par,DecStat,p,4,4)
    #      s_out[start:start+Frame] = fs_out
    #      start = start + Frame
    #
    #  wavfile.write("s_out.wav", 16000, np.int16(s_out*2.**15))
