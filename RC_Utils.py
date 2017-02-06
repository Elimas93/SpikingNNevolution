"""
Contains utility functions for RC simulations
"""

import numpy as np

def createInputSine(input_freq, input_duration, sample_freq):
    """
    :param input_freq: freq of input sine, Hz
    :param input_duration: in s
    :param sample_freq: in Hz
    :return:
    """
    n_samples = input_duration * sample_freq
    w = 2. * np.pi * input_freq
    time = np.linspace(0, input_duration, n_samples)
    y = np.sin(w * time)
    return y, time