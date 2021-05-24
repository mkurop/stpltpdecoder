# -*- coding: utf-8
 
__author__ = 'Marcin Kuropatwiński'

"""Module for the computation of the Short Term Predictor and the Long Term Predictor parameters. \
Provided classes for encoder and decoder work for any specified sampling frequency.

Created 31.03.2014 Code modernized 22.05.2021"""

import os
from typing import Tuple, List

import numpy as np
from scipy.io import wavfile
from scipy.signal import deconvolve, lfilter, lfiltic

from spectrum import *


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

        # order of the STP filter

        self.p = 10 if sampling_rate == 8000 else 16 

        # length of the signal frame in samples

        self.frame = 160 if sampling_rate == 8000 else 320
        
        # number of subframes in a frame

        self.subframes = 4

        # length of the signal subframe in samples

        self.subframe = self.frame / self.subframes

        # minimal and maximal pitch lag
        self.pitch_lag_max = 120 * sampling_rate/8000
        self.pitch_lag_min = 20 * sampling_rate/8000

        # parameters for the poles bandwidth widening procedure
        self.g1 = 0.994
        self.gk1 = np.cumprod(np.ones((self.p,)) * g1)


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

            short_term_res = np.random.randn((self.cfg.pitch_lag_max,))*1e-6 # initial conditions for the LTP analysis


    def __init__(self, sampling_rate: int = 16000):
        """Constructor method"""

        # instantiate object of type Config
        
        self.cfg = Config(sampling_rate)

        # instantiate encoder state object

        self.state = EncoderState(sampling_rate)

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
        a, stp_excitation_variance, reflection_coefficients = levinson(x[p:]) 

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

        short_term_residual = np.concatenate(self.state.short_term_res,current_frame_short_term_res)

        current_frame_long_term_res = np.zeros_like(current_frame_short_term_res)

        # space for pitches in subframes (LTP lags)
        ltp_lags = np.zeros((self.cfg.subframes,))

        # space for LTP taps
        ltp_taps = np.zeros_like(ltp_lags)

        # space for LTP variances
        ltp_variances = np.zeros_like(ltp_lags)
        
        for i in range(self.cfg.subframes):

            current_subframe = current_frame_short_term_res[i*self.cfg.subframe:(i+1)*self.cfg.subframe]

            past_short_term_residual = short_term_residual[i*self.cfg.subframe: (i+i)*self.cfg.subframe + self.cfg.pitch_lag_max - self.cfg.pitch_lag_min]

            # long term correlations
            aux_matrix1 = np.expand_dim(np.arange(self.cfg.subframe,0)) + np.expand_dim(np.arange(self.cfg.pitch_lag_max-self.cfg.pitch_lag_min),0).T

            aux_matrix2 = past_short_term_residual[aux_matrix1] 

            long_term_correlations = np.sum(aux_matrix2 * current_subframe,axis=1)

            denominators = np.sum(aux_matrix2 * aux_matrix2,axis=1)

            merit = long_term_correlations/denominators

            aux_matrix3 = merit * long_term_correlations

            aux1 = np.argmax(aux_matrix3)

            ltp_lags[i] = self.cfg.pitch_lag_max - aux1
            
            ltp_taps[i] = -merit[aux1]

            ltp_variances[i] = np.sum(current_subframe**2) - aux_matrix3[aux1]

            current_frame_long_term_res[i*self.cfg.subframe:(i+1)*self.cfg.subframe] = (current_subframe + ltp_taps[i]*aux_matrix[aux1,:].ravel())/np.sqrt(ltp_variances[i])

        # update state for next frame
        self.state.lsf = lsf
        self.state.short_term_res = current_frame_short_term_res[-self.cfg.pitch_lag_max:]
        
        # fill output parameters structure
        output_parameters.lsf = lsf_interpolated
        output_parameters.a = a_interpolated
        output_parameters.ltp_lags = ltp_lags
        output_parameters.ltp_taps = ltp_taps
        output_parameters.ltp_variances = ltp_variances
        output_parameters.current_frame_long_term_res = current_frame_long_term_res

        return output_parameters


class StpLetDecoder:

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

            short_term_res = np.random.randn((self.cfg.pitch_lag_max,)))*1e-6

    def __init__(self, sampling_rate : int):

        self.cfg = Config(sampling_rate : int =16000)

        self.state = DecoderState(sampling_rate)

    def frame(parameters) -> np.ndarray:
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

        output_signal_frame = np.zeros((self.cfg.frame,))

        for i in range(self.cfg.subframes):

            # LTP synthesis -------------------------------

            short_term_residual = np.sqrt(parameters.ltp_variances[i]) * parameters.current_frame_long_term_res[i*self.cfg.subframe:(i+1)*self.cfg.subframe]

            beg1 = self.cfg.pitch_lag_max-parameters.ltp_lags[i]

            end1 = np.min(self.cfg.pitch_lag_max, beg1 + self.cfg.subframe)

            aux_vector1 = parameters.ltp_taps[i] * self.state.short_term_res[beg1:end1]

            aux1 = end1 - beg1

            short_term_residual[:aux1] += aux_vector1

            beg2 = aux1 - parameters.ltp_lags[i]

            end2 = self.cfg.subframe - parameters.ltp_lags[i]

            if aux1 < self.cfg.subframe:

                # extend current frame short_term_residual with the past samples of the STP residual
                short_term_residual = np.concatenate(self.state.short_term_res,short_term_residual)
                
                aux2 = self.cfg.pitch_lag_max + aux1

                len1 = self.cfg.subframe - aux1

                beg2 = self.cfg.pitch_lag_max + aux1-parameters.ltp_lags[i]

                end2 = beg2 + len1 

                short_term_residual[aux2:] += parameters.ltp_taps[i]*short_term_residual[beg2:end2]

            #  if beg2 < 0:
            #
            #      len1 = -beg2
            #
            #      aux2 = np.min(self.cfg.subframe,aux1+len1)
            #
            #      short_term_residual[aux1:aux2] = parameters.ltp_taps[i] * self.state.short_term_res[self.cfg.pitch_lag_max + beg2:]
            #
            #      len3 = self.cfg.subframe - aux2
            #
            #      beg3 = aux2 - parameters.ltp_lags[i]
            #
            #      end3 = beg3 + len3
            #
            #      short_term_residual[aux2:] = parameters.ltp_taps[i] * short_term_residual[beg3:end3]
            #
            #  else:
            #
            #      short_term_residual[aux1:] = parameters.ltp_taps[i] * short_term_residual[beg2:end2]

            # update state

            self.state.short_term_res = short_term_residual[self.cfg.subframe:] 
            
            # STP synthesis ----------------------------------

            # recreate the output frame and update initial conditions for the STP synthesis filter

            [output_signal_frame[i*self.cfg.subframe:(i+1)*self.cfg.subframe], self.state.hlp] = lfilter([1.0],parameters.a[:,i].ravel(),short_term_residual[self.cfg.pitch_lag_max:],zi = self.state.hlp)

        return output_signal_frame

def wav_files_in_directory(dir : str) -> List[str]:
    """ Finds all files with the .wav extention in subdirectories of the dir folder.

    :param dir: the folder to be searched over
    :type dir: str

    :return: the list of files with .wav extension found
    :rtype: List[str]

    :raises: 
    """
    
    if not os.path.exists(dir):
        raise ValueError(f"The directory {dir} does not exists.")

    files_list = []

    for roots,dirs,files in os.walk(dir):

        for file_ in files:

            if file_.endswith("WAV") or file_.enswith("wav"):

                files_list.append(file_)

    return files_list


if __name__ == "__main__":
    sr, s = wavfile.read("VLRecording10591_05.04.13_25.168.wav")
    s = s/2.**15
    s_out = np.zeros_like(s)
    CodStat = InitEncStat(16, 512)
    DecStat = InitDecStat(16, 512)
    start = 0
    Frame = 512
    p = 16
    while start + Frame < len(s):
        fs = s[start:start+Frame]
        CodStat, Par, LongTermRes = Encoder(fs, CodStat, p, Frame, 4, 4, 16000)
        fs_out, DecStat = Decoder(LongTermRes,Par,DecStat,p,4,4)
        s_out[start:start+Frame] = fs_out
        start = start + Frame

    wavfile.write("s_out.wav", 16000, np.int16(s_out*2.**15))
