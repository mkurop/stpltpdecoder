# -*- coding: utf-8
 
__author__ = 'Marcin Kuropatwiński'

"""Module for the computation of the Short Term Predictor and the Long Term Predictorparameters. \
Provided classes for encoder and decoder work for any specified sampling frequency.

Created 31.03.2014 Code modernized 22.05.2021"""

from typing import Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import deconvolve

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
        """Class holding the state of the encoder, which changes from frame to frame

        :param sampling_rate: sampling rate in Hz of the signal processed
        :type sampling_rate: int
        """

        def __init__(self, sampling_rate: int):
            """Constructor method"""

            self.cfg = Config(sampling_rate)

            self.reset_state()

        def reset_state(self):

            self.lsf = np.linspace(0.2, np.pi - 0.1, self.cfg.p) 

            self.hlp = np.zeros((self.cfg.p,)) 

            short_term_res = np.zeros((self.cfg.pitch_lag_max,)))


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
            * lsf - the line spectral frequency parameters
            * a - the STP polynomial coefficients
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
        output_parameters.lsf = lsf
        output_parameters.a = a
        output_parameters.ltp_lags = ltp_lags
        output_parameters.ltp_taps = ltp_taps
        output_parameters.ltp_variances = ltp_variances
        output_parameters.current_frame_long_term_res = current_frame_long_term_res

        return output_parameters



























def InitDecStat(p,Frame):
    """Function initializing the decoder
    Input:
    p - AR model order
    Frame - length of the frame
    Author: Marcin Kuropatwiński
    Created: 31.03.2014
    """
    DecStat = col.namedtuple('DecStat','LsfQ, hlp, ShortTermRes')
    DecStat = DecStat(LsfQ = np.linspace(0.2,np.pi-0.1,p), hlp = np.zeros((p,)), ShortTermRes = np.zeros((Frame,)))
    return DecStat


def Decoder(LongTermRes,Par,DecStat,p,SubFrames,SubFramesOL):

    """
    Function for decoding speech based on the input from the complementary function Encoder.

    Input:
    LongTermRes - a frame of the long term residual
    Par - parameters structure
    DecState - structure of the decoder state with the following fields

    hlp - short term residual synthesis filter state
    ShortTermRes - previous frame of the short term residual

    Remaining inputs are obvious, see the Encoder function.

    Output:

    fs - current frame synthesized speech
    DecState - decoder state structure

    Author: Marcin Kuropatwinski
    Created: 31.03.2014
    """
    #declare output DecStat
    DecStatOut = col.namedtuple('DecStatOut','LsfQ, hlp, ShortTermRes')

    #length of the step
    Frame = len(LongTermRes)
    #number of subframes per frame
    n = SubFrames
    #length of the subframe
    sfl = Frame/n
    np_ = SubFramesOL # number of open loop frames
    # length of the open loop pitch subframe
    sflp = Frame/np_

    # get the current frame parameters
    LsfQ = Par.LsfQ
    LtpLag = Par.LtpLag
    LtpTap = Par.LtpTap
    LtpVar = Par.LtpVar
    hlp = copy.copy(DecStat.hlp)


    #resynthesise speech
    #initial state of the long term synthesis filter - length of 2 frames
    ssr = np.concatenate([DecStat.ShortTermRes,np.zeros((Frame,))],axis=0)

    #resynthesise frame of the short term residual
    start = 0
    for i in range(np_):
        #    print len(LtpLag), len(LtpVar), len(LtpTap)
        for j in range(i*sflp,(i+1)*sflp):
            #raw_input()
            ssr[Frame+j] = np.sqrt(LtpVar[i])*LongTermRes[start+j] - \
            LtpTap[i]*ssr[Frame+j-LtpLag[i]]

    ShortTermRes = ssr[Frame:]
    aslpq = DecStat.LsfQ
    aslq = np.array(LsfQ,ndmin=1)

    LsfQ = aslq
    ic = 1./n
    fs = np.zeros((Frame,))
    for i in range(n):
        asli = (1-(i+1)*ic)*aslpq + (i+1)*ic*aslq
        # convert back into lpc domain
        a = lsf2poly(asli)
        for j in range(i*sfl,(i+1)*sfl):
            fs[j] = -np.sum(a[1:]*hlp) + ssr[Frame+j]
            hlp[1:] = hlp[:-1]
            hlp[0] = fs[j]

    DecStatOut = DecStatOut(LsfQ = LsfQ, hlp = hlp, ShortTermRes = ShortTermRes)

    return fs, DecStatOut

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
