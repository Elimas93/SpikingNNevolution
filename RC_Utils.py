"""
Contains utility functions for RC simulations
"""

import numpy as np
import scipy.interpolate as spint
import sklearn.linear_model

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

def createInputSpline(input_freq, input_duration, sample_freq,Y0,Y1,X1):

    samplesPerPeriod = int(round(sample_freq/input_freq))
    samples = int(round(sample_freq*input_duration))

    x = [0.,X1,1.]
    y = [Y0,Y1,Y0]
    Interpolator = spint.PchipInterpolator(x, y)

    xspline = np.linspace(0.,1.,samplesPerPeriod+1)
    yspline = Interpolator.__call__(xspline)

    target = []
    for i in range(samples):
        i=i%samplesPerPeriod
        target.append(yspline[i])

    return target

def doRidgeRegression(X_train,X_run,Y_train,Y_run,alphas):
    """
    :param X_train:
    :param X_run:
    :param Y_train:
    :param Y_run:
    :param alphas: list, ridge regression parameter
    :return:
    """
    # do ridge regression
    w_out_ridge = []
    readout_train_list = []
    readout_list = []
    mse_list = []
    for alpha in alphas:
        ridge_regressor = sklearn.linear_model.Ridge(alpha, fit_intercept=False)
        rr = ridge_regressor.fit(X_train, Y_train)
        # rr = ridge_regressor.fit(X, np.transpose(inputs))  # identity prediction
        w_out_ridge.append(np.transpose(rr.coef_))  # store weights

        readout_train = rr.predict(X_train)
        readout_train_list.append(readout_train)

        # apply weights to run data
        readout = rr.predict(X_run)
        readout_list.append(readout)

        # calculate MSE
        mse = ((Y_run - readout) ** 2).mean()
        mse_list.append(mse)
    return w_out_ridge, readout_list, readout_train_list, mse_list

