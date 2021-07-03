#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Marcin Kuropatwiński"

__all__ = ['lpanafil', 'wav_files_in_directory', 'clear_output_directory', 'frames_generator', 'read_wav_and_normalize', 'write_wav' ]
import os
import shutil
from typing import Tuple, List
from scipy.io import wavfile
import numpy as np

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

    s += np.random.randn(*s.shape)*1.e-4 # add dither to improve numerical behaviour

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
