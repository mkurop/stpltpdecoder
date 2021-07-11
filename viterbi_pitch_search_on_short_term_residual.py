#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from typing import Tuple, Dict, List
from numba import njit, uint32, float32
from numba.experimental import jitclass
import numba.typed 

import os
import shutil

from scipy.io import wavfile
from scipy.signal import deconvolve, lfilter, lfiltic
from decoder import *
from spectrum_decoder import *
from utils_decoder import *

spec_hypothesis = [
    ('_ltp_lag', uint32),
    ('_ltp_tap', float32),
    ('_ltp_variance', float32),
    ('_backpointer', uint32),
    ('_cost', float32),
]

FIRST_SUBFRAME_BACKPOINTER = 1000

@jitclass(spec_hypothesis)
class Hypothesis:

    def __init__(self, ltp_lag, ltp_tap, ltp_variance, backpointer, sum_of_ltp_variances_until_given_subframe):

        self._ltp_lag = ltp_lag

        self._ltp_tap = ltp_tap

        self._ltp_variance = ltp_variance

        self._backpointer = backpointer

        self._cost = sum_of_ltp_variances_until_given_subframe

    @property
    def ltp_lag(self):

        return self._ltp_lag

    @property
    def ltp_tap(self):

        return self._ltp_tap

    @property
    def ltp_variance(self):

        return self._ltp_variance

    @property
    def backpointer(self):

        return self._backpointer

    @property
    def cost(self):

        return self._cost

    @ltp_lag.setter
    def ltp_lag(self, ltp_lag_):

        self._ltp_lag = ltp_lag_

    @ltp_tap.setter
    def ltp_tap(self, ltp_tap_):

        self._ltp_tap = ltp_tap_

    @ltp_variance.setter
    def ltp_variance(self, ltp_variance_):

        self._ltp_variance = ltp_variance_

    @backpointer.setter
    def backpointer(self, backpointer_):

        self._backpointer = backpointer_

    @cost.setter
    def cost(self, sum_of_ltp_variances_until_given_subframe):

        self._cost = sum_of_ltp_variances_until_given_subframe


spec_viterbi_ltp_parameters_trajectories = [
    ('_short_term_residual', float32[:]),
    ('_r_i', float32[:]),
    ('_full_short_term_residual', float32[:]),
    ('_L_min', uint32),
    ('_L_max', uint32),
    ('_fan_out', uint32),
    ('_frame', uint32),
    ('_subframes', uint32),
    ('_subframe_length', uint32),
    ('_n_best', uint32),
]

@jitclass(spec_viterbi_ltp_parameters_trajectories)
class ViterbiLtpParametersTrajectories:
    """ViterbiLtpParametersTrajectories."""

    def __init__(self, L_max : int = 240, L_min : int = 40, fan_out : int = 9, n_best : int = 5):
        """Inits the object with the short_term_residual in the current frame and r_i - the short term residual initial conditions vector of length L_max,\
        where L_max is the maxium long term predictor lag possible.
        :param L_max: maximal allowed pitch lag
        :param L_min: minimal allowed pitch lag
        :type L_min: int
        :param fan_out: the next frame pich is L +/- fan_out//2 where L is the previous frame pitch
        :type fan_out: int
        :param n_best: retuns n_best pitch tracks
        :type n_best: int
        """

        self._L_max = L_max
        self._L_min = L_min
        self._fan_out = fan_out
        self._subframes = 4 # number of subframes in a frame
        self._n_best = n_best

        self._r_i = (np.random.randn(self._L_max,)*1e-8).astype(np.float32)

    def ltp_tap_and_ltp_variances_for_given_ltp_lag_and_subframe(self, ltp_lag : int, subframe : int) -> Tuple[float,float]:

        short_term_residual_subframe = self._short_term_residual[subframe*self._subframe_length:(subframe+1)*self._subframe_length]


        ltp_lag_delayed_short_term_residual_subframe = self._full_short_term_residual[self._L_max+subframe*self._subframe_length-ltp_lag:self._L_max + subframe*self._subframe_length-ltp_lag+self._subframe_length]

        
        #  input('PE')

        cross_correlation = np.sum(short_term_residual_subframe*ltp_lag_delayed_short_term_residual_subframe)

        delayed_subframe_energy = np.sum(ltp_lag_delayed_short_term_residual_subframe*ltp_lag_delayed_short_term_residual_subframe)

        ltp_tap = -cross_correlation/delayed_subframe_energy

        subframe_energy = np.sum(short_term_residual_subframe*short_term_residual_subframe)

        ltp_variance = (subframe_energy - cross_correlation*cross_correlation/delayed_subframe_energy)/self._subframe_length

        return ltp_tap, ltp_variance

    def list_of_lags_for_first_subframe(self) -> np.ndarray:

        return np.arange(np.int32(self._L_min),np.int32(self._L_max+1))

    def list_of_lags_for_next_subframe(self, previous_subframe_lag_list : np.ndarray) -> np.ndarray:
        """Lists unique ltp lags for the next, after first, subframe 

        :param previous_subframe_lag_list: unique ltp lags from previous subframe
        :type previous_subframe_lag_list: np.ndarray
        :return: unique ltp lags for the current subframe
        :rtype: np.ndarray
        """

        output_lags = []

        for ltp_lag in previous_subframe_lag_list:

            for next_ltp_lag in range(max(ltp_lag-self._fan_out//2,self._L_min),min(ltp_lag+self._fan_out//2,self._L_max)+1):

                output_lags.append(next_ltp_lag)

        return np.unique(np.asarray(output_lags))

    def dict_of_lags_to_taps_and_vars(self, lag_list : np.ndarray, subframe : int) -> Dict[int, Tuple[float,float]]:

        out_dict = {}

        for ltp_lag in lag_list:

            out_dict[ltp_lag] = self.ltp_tap_and_ltp_variances_for_given_ltp_lag_and_subframe(ltp_lag, subframe)

        return out_dict

    def initialize_hypotheses_list_for_first_subframe(self) -> List[Hypothesis]:

        lag_list = self.list_of_lags_for_first_subframe()

        lag_to_tap_var_dict = self.dict_of_lags_to_taps_and_vars(lag_list, 0)

        hypotheses_list = numba.typed.List() 

        for lag, tap_var in lag_to_tap_var_dict.items():

            tap = tap_var[0]

            var = tap_var[1]

            hypotheses_list.append(Hypothesis( lag, tap, var, FIRST_SUBFRAME_BACKPOINTER, var))

        return hypotheses_list

    def expand_for_other_than_first_subframe(self, hypotheses_list : List[Hypothesis], dict_of_lags_to_taps_and_vars_ : Dict[int, Tuple[float,float]]) -> List[Hypothesis]:

        TAP_INDEX = 0
        VAR_INDEX = 1

        new_hypotheses_list = numba.typed.List() 

        for backpointer, hypothesis in enumerate(hypotheses_list):

            for lag in range(max(hypothesis.ltp_lag-self._fan_out//2, self._L_min),min(hypothesis.ltp_lag+self._fan_out//2, self._L_max)+1):
                tap = dict_of_lags_to_taps_and_vars_[lag][TAP_INDEX]
                var = dict_of_lags_to_taps_and_vars_[lag][VAR_INDEX]
                new_hypotheses_list.append(Hypothesis( lag, tap, var, backpointer, hypothesis.cost + var))

        return new_hypotheses_list

    def sort_hypotheses_list_wrt_cost(self, hypotheses_list : List[Hypothesis]) -> List[Hypothesis]:

        costs_list = numba.typed.List() 

        for hypothesis in hypotheses_list:

            costs_list.append(hypothesis.cost)

        costs_list_ndarray = np.asarray(costs_list)

        indices = np.argsort(costs_list_ndarray)

        # form output list using list comprehension

        output_hypotheses_list = [hypotheses_list[i] for i in indices]

        return output_hypotheses_list

    def histogram_pruning(self, hypotheses_list : List[Hypothesis], pruning : int = 10):

        hypotheses_list_sorted = self.sort_hypotheses_list_wrt_cost(hypotheses_list)

        return hypotheses_list_sorted[:min(pruning,len(hypotheses_list))]

    def get_dict_of_lags_to_taps_and_vars_for_the_next_subframe(self, hypotheses_list_pruned: List[Hypothesis], subframe : int) -> np.ndarray:

        aux0 = set()

        for hypothesis in hypotheses_list_pruned:

            aux0.add(hypothesis.ltp_lag)

        lags_list = np.asarray(list(aux0))

        list_of_lags_for_next_subframe_ = self.list_of_lags_for_next_subframe(lags_list)

        dict_lags_to_taps_and_vars_ = self.dict_of_lags_to_taps_and_vars(list_of_lags_for_next_subframe_, subframe)

        return dict_lags_to_taps_and_vars_

    def frame(self, short_term_residual_frame : np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        self._frame = short_term_residual_frame.shape[0]

        self._short_term_residual = short_term_residual_frame.astype(np.float32)

        self._full_short_term_residual = np.concatenate((self._r_i, self._short_term_residual))

        self._subframe_length = self._frame // self._subframes

        # first subframe

        hypotheses_list_1st = self.initialize_hypotheses_list_for_first_subframe()

        # histogram pruning

        hypotheses_list_1st_pruned = self.histogram_pruning(hypotheses_list_1st)

        # get dict for the 2nd subframe

        subframe = 1

        dict_lags_to_taps_and_vars_ = self.get_dict_of_lags_to_taps_and_vars_for_the_next_subframe(hypotheses_list_1st_pruned, subframe)

        # expand for the list of hypotheses in the 2nd subframe

        hypotheses_list_2nd = self.expand_for_other_than_first_subframe(hypotheses_list_1st_pruned, dict_lags_to_taps_and_vars_)

        # histogram pruning in the 2nd subframe

        hypotheses_list_2nd_pruned = self.histogram_pruning(hypotheses_list_2nd)

        # get dict for the 3rd subframe

        subframe = 2

        dict_lags_to_taps_and_vars_ = self.get_dict_of_lags_to_taps_and_vars_for_the_next_subframe(hypotheses_list_2nd_pruned, subframe)

        # expand for the list of hypotheses in the 3rd subframe

        hypotheses_list_3rd = self.expand_for_other_than_first_subframe(hypotheses_list_2nd_pruned, dict_lags_to_taps_and_vars_)

        # histogram pruning in the 3rd subframe

        hypotheses_list_3rd_pruned = self.histogram_pruning(hypotheses_list_3rd)

        # get dict for the 4th subframe

        subframe = 3

        dict_lags_to_taps_and_vars_ = self.get_dict_of_lags_to_taps_and_vars_for_the_next_subframe(hypotheses_list_3rd_pruned, subframe)

        # expand for the list of hypotheses in the 4th subframe

        hypotheses_list_4th = self.expand_for_other_than_first_subframe(hypotheses_list_3rd_pruned, dict_lags_to_taps_and_vars_)

        # histogram pruning in the 4th subframe

        hypotheses_list_4th_pruned = self.histogram_pruning(hypotheses_list_4th)

        ltp_lags = np.zeros((self._n_best, self._subframes))

        ltp_taps = np.zeros((self._n_best, self._subframes))

        ltp_variances = np.zeros((self._n_best, self._subframes))

        costs = np.zeros((self._n_best,))

        for i in range(self._n_best):

            backpointer_4th = i
            backpointer_3rd = hypotheses_list_4th_pruned[backpointer_4th].backpointer
            backpointer_2nd = hypotheses_list_3rd_pruned[backpointer_3rd].backpointer
            backpointer_1st = hypotheses_list_2nd_pruned[backpointer_2nd].backpointer

            ltp_lag_1st = hypotheses_list_1st_pruned[backpointer_1st].ltp_lag
            ltp_lag_2nd = hypotheses_list_2nd_pruned[backpointer_2nd].ltp_lag
            ltp_lag_3rd = hypotheses_list_3rd_pruned[backpointer_3rd].ltp_lag
            ltp_lag_4th = hypotheses_list_4th_pruned[backpointer_4th].ltp_lag

            ltp_lags[i,:] = np.asarray([ltp_lag_1st, ltp_lag_2nd, ltp_lag_3rd, ltp_lag_4th])

            ltp_tap_1st = hypotheses_list_1st_pruned[backpointer_1st].ltp_tap
            ltp_tap_2nd = hypotheses_list_2nd_pruned[backpointer_2nd].ltp_tap
            ltp_tap_3rd = hypotheses_list_3rd_pruned[backpointer_3rd].ltp_tap
            ltp_tap_4th = hypotheses_list_4th_pruned[backpointer_4th].ltp_tap

            ltp_taps[i,:] = np.asarray([ltp_tap_1st, ltp_tap_2nd, ltp_tap_3rd, ltp_tap_4th])

            ltp_variance_1st = hypotheses_list_1st_pruned[backpointer_1st].ltp_variance
            ltp_variance_2nd = hypotheses_list_2nd_pruned[backpointer_2nd].ltp_variance
            ltp_variance_3rd = hypotheses_list_3rd_pruned[backpointer_3rd].ltp_variance
            ltp_variance_4th = hypotheses_list_4th_pruned[backpointer_4th].ltp_variance

            ltp_variances[i,:] = np.asarray([ltp_variance_1st, ltp_variance_2nd, ltp_variance_3rd, ltp_variance_4th])

            costs[i] = hypotheses_list_4th_pruned[backpointer_4th].cost

        self._r_i = self._short_term_residual[-self._L_max:]

        # compute long term residual for the best track

        excitation = np.zeros((self._frame,),dtype=np.float32)

        for i in range(self._subframes):

            excitation[i*self._subframe_length:(i+1)*self._subframe_length] = self._full_short_term_residual[self._L_max+i*self._subframe_length:self._L_max+(i+1)*self._subframe_length] + ltp_taps[i] * self._full_short_term_residual[self._L_max-ltp_lags[i] + i*self._subframe_length:self._L_max-ltp_lags[i]+(i+1)*self._subframe_length]/np.sqrt(ltp_variances[i])

        return ltp_taps, ltp_lags, ltp_variances, excitation, costs

if __name__ == "__main__":

    files = wav_files_in_directory('./data/input/')

    sample_rate = 16000

    for file_ in files:

        speech_signal, sample_rate = read_wav_and_normalize(file_)

        encode = StpLtpEncoder(sample_rate, only_stp_analysis = False, return_short_term_residual = True)

        ltp_viterbi = ViterbiLtpParametersTrajectories()

        for signal_frame in frames_generator(speech_signal, encode.cfg.frame):

            params = encode.frame(signal_frame)

            print(params.current_frame_short_term_res.shape)

            ltp_taps, ltp_lags, ltp_variances, costs = ltp_viterbi.frame(params.current_frame_short_term_res)

            params = encode.frame(signal_frame)

            print(params.current_frame_short_term_res.shape)

            ltp_taps1, ltp_lags1, ltp_variances1, costs1 = ltp_viterbi.frame(params.current_frame_short_term_res)

            print(ltp_lags)
            print(ltp_taps)
            print(ltp_variances)
            print(costs)

            print(params.ltp_lags)

            input("PE")




