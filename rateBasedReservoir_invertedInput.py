"""
simulation of rate based reservoir
training with linear regression

fraction of inputs can be inverted (trick to make reservoir dynamics more interesting)
"""


import plotly.plotly as plotly
import plotly.graph_objs as graph_objs
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn.linear_model

from IPython.core.debugger import Tracer


class Neuron(object):
    def __init__(self):
        "Nothing to do here"

    def tran_fct(self, xStates):
        "Neuron activation function"

        x = xStates.copy()
        for idx,itm in enumerate(x):
            if itm <= 0 :
                x[idx] = 0
            else :
                x[idx] = 1.05*itm/(1.6+itm)
        return x

        # return np.tanh(xStates)

class ReservoirNet(Neuron):
    """
    This class implements constant and methods of a artificial neural network
    using Reservoir Computing architecture, with the goal of comparing with a network of spiking populations.
    """

    def __init__(self, n_in=0, n_out=1, n_res=100, spec_rad=1.15, leakage=0.1, scale_bias=0.5, scale_fb=5.0, scale_noise=0.01, scale_fb_noise=0.01,
                 verbose=False,negative_weights=True,fraction_inverted=0.0):
        "Initialize weights of the reservoir"

        Neuron.__init__(self)

        self.scale_bias = scale_bias
        self.scale_fb = scale_fb
        self.scale_noise = scale_noise
        self.scale_feedback_noise = scale_fb_noise

        self.leakage = leakage

        self.TRANS_PERC = 0.1

        self.n_in = n_in
        self.n_out = n_out
        self.n_res = n_res

        self.w_out = np.random.randn(n_res, n_out)
        if not (negative_weights):
            self.w_out = abs(self.w_out)
        self.w_in = np.random.randn(n_res, n_in)
        if not(negative_weights):
            self.w_in = abs(self.w_in)
        self.w_bias = np.random.randn(n_res,1) * self.scale_bias
        self.w_bias = 0.05 #mimics resting noise level of approx 50 Hz
        self.w_res = self.get_rand_mat(n_res, spec_rad,negative_weights=negative_weights) # (to,from)

        self.w_res += 0.05

        self.w_fb = np.random.randn(n_res, n_out) * self.scale_fb
        if not(negative_weights):
            self.w_fb = abs(self.w_fb)
        self.N_inverted = np.round(n_res*fraction_inverted)
        # set initial state
        # self.x = np.random.randn(n_res, 1)
        self.x = np.zeros((n_res, 1))
        self.u = np.zeros(n_in)
        self.y = np.zeros(n_out)

        self.verbose = verbose

    def get_rand_mat(self, dim, spec_rad,negative_weights=True):
        "Return a square random matrix of dimension @dim given a spectral radius @spec_rad"

        mat = np.random.randn(dim, dim)
        if not(negative_weights):
            mat = abs(mat)
        w, v = np.linalg.eig(mat)
        mat = np.divide(mat, (np.amax(np.absolute(w)) / spec_rad))

        return mat

    def update_state(self, u=0, y=0):
        """
        Update the state of the reservoir
        u = input vector
        y = output vector (can be used for output feedback)
        """

        #calculate (inverted) feedback contribution
        N_normal = self.n_res - self.N_inverted
        fb1 = self.w_fb[:N_normal, :] * (np.ones((N_normal, 1)) * y + np.random.randn(N_normal, 1) * self.scale_feedback_noise)  # w_fb*(y+noise)
        fb2 = self.w_fb[N_normal:, :] * (np.ones((self.N_inverted, 1)) * (0.6 - y) + np.random.randn(self.N_inverted,
                                                                                                   1) * self.scale_feedback_noise)  # w_fb*(1-y+noise)
        fb = np.vstack((fb1, fb2))

        #create noise term
        noise = np.random.randn(self.n_res, 1) * self.scale_noise

        # Reservoir update equation if no input
        if self.n_in == 0:
            # x_new = np.dot(self.w_res, self.x) + self.w_bias + np.dot(self.w_fb, y)
            x_new = np.dot(self.w_res, self.x) + self.w_bias + fb + noise

        # Reservoir update equation if input
        else:
            x_new = np.dot(self.w_res, self.x) + self.w_bias + fb + np.dot(self.w_in, u) + noise
        # leakage

        self.x = (1 - self.leakage) * self.x + self.leakage * self.tran_fct(x_new)

        if max(self.x) > 500:
            Tracer()()

    def run(self, n_it, U=None, fakeFB=None, to_plot=200):
        """
        Run the network for n_it timesteps given a timeserie of input vector U . \
        U = inputs with shape (n_it,n_in)
        fakeFB = np array, will be used instead of the actual network feedback (useful to test attractor state strength)
        """

        #disable (feedback) noise
        # self.scale_feedback_noise = 0.0
        # self.scale_noise = 0.0
        self.scale_noise = self.scale_noise/2

        Y = np.array([])
        X = np.array([])
        if self.n_in > 0:
            if U.shape != (n_it,self.n_in):
                raise ValueError("inputs should be of shape (n_it,n_in)")

        if fakeFB == None:
            num_fb = 0
        else:
            num_fb = fakeFB.shape[0]

        print("\n -- RUNNING --")
        print(" -- Update the state matrix for " + str(n_it) + " time steps -- ")
        for j in range(n_it):

            if j < num_fb:
                y = fakeFB[j][0]
            elif j == 0:
                y = 0.0
            else:
                y = Y[j - 1, :].reshape(-1, 1)

            if self.n_in == 0:
                    self.update_state(y=y) # feed back the output of previous timestep/fakeFB
            else:
                    self.update_state(u=U[j].reshape(-1, 1), y=y) # feed back the output of previous timestep/fakeFB

            y = np.dot(np.transpose(self.w_out),self.x)
            Y = np.vstack([Y, np.transpose(y)]) if Y.size else np.transpose(y)
            X = np.vstack([X, np.transpose(self.x)]) if X.size else np.transpose(self.x)

        self.X = X
        self.Y = Y

        if self.X.min() == 0 and self.X.max() == 0:
            print "!!! WARNING: it seems there has been no activity whatsoever in the network !!!"

        # p = Plotting(X)
        # p.plot_3d(to_plot, False)

        plt.figure()
        [plt.plot(X[:, x]) for x in range(X.shape[1])]
        plt.title("rate-based Neuron states during run")
        plt.xlabel("timestep")
        plt.ylabel("Neuron state")
        # plt.ylim(0,1)
        plt.show()
        # Tracer()()

    def train(self, Ytrain, U=np.zeros([]), to_plot=200):
        "Train the network given a timeserie of input vector @U and desired output vector @Y. \
        To perform the training, the network is first updated for each input and a matrix \
        X of the network states is sampled. After removing the transiant states, \
        w_out is computed with least square"

        n_it = Ytrain.shape[0] - 1
        X = np.array([])
        if self.n_in > 0:
            if U.shape != (Ytrain.shape[0],self.n_in):
                raise ValueError("inputs should be of shape (Ntrainingsamples,n_in)")

        print("\n -- RUNNING to generate training data --")
        print(" -- Update the state matrix for " + str(n_it) + " time steps -- ")
        for j in range(n_it):
            if self.n_in == 0:
                self.update_state(y=Ytrain[j,:].reshape(-1, 1))  # feed back the desired output of previous timestep

            else:
                self.update_state(u=U[j].reshape(-1, 1), y=Ytrain[j,:].reshape(-1, 1))  # feed back the desired output of previous timestep

            # y = np.dot(np.transpose(self.w_out),self.x)
            # Y = np.vstack([Y, np.transpose(y)]) if Y.size else np.transpose(y)
            X = np.vstack([X, np.transpose(self.x)]) if X.size else np.transpose(self.x)


        if X.min() == 0 and X.max() == 0:
            print "!!! WARNING: it seems there has been no activity whatsoever in the network !!!"

        # p = Plotting(X)
        # p.plot_3d(to_plot, False)

        plt.figure()
        [plt.plot(X[:, x]) for x in range(X.shape[1])]
        plt.title("rate-based Neuron states during train")
        plt.xlabel("timestep")
        plt.ylabel("Neuron state")
        # plt.ylim(0,1)
        plt.show()

        # store X and Y
        # Tracer()()
        self.X = X.copy()
        self.Y = Ytrain.copy()

        print(" -- Removing transiant states -- ")
        to_remove = int(self.TRANS_PERC * n_it)
        X = np.delete(X, np.s_[0:to_remove], 0)
        Ytrain = np.delete(Ytrain, np.s_[0:to_remove], 0)


        # plt.figure()
        # [plt.plot(X[:, x]) for x in range(X.shape[1])]
        # plt.title("Neuron states during training")
        # plt.xlabel("timestep")
        # plt.ylabel("Neuron state")

        print(" -- Updating w_out using linear regression -- ")
        if self.verbose == True:
            print("( inv( " + str(np.transpose(X).shape) + " X " + str(X.shape) + " ) X " + str(np.transpose(X).shape) + \
                  " ) X " + str(Ytrain.shape))

        # self.w_out = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), Ytrain[1:])
        if self.x.shape[1] > 1:
            Tracer()()


        # do ridge regression
        fig = plt.figure()
        plt.plot(Ytrain,color='blue',linewidth=2)
        # powers = np.hstack((-10., np.arange(-9.,-7.,0.25,)))
        # powers = np.hstack((powers, np.arange(-7,-2)))
        # powers = np.arange(-3.,1.)
        powers = [-3.,-2.,-1.3,-1.,0.]
        self.alphas = [ np.power(10,x) for x in powers]
        # self.alphas = [0.0001]
        self.w_out_ridge = []
        for alpha in self.alphas:
            ridge_regressor = sklearn.linear_model.Ridge(alpha,fit_intercept=False)
            # Tracer()()
            rr = ridge_regressor.fit(X, Ytrain[1:])
            readout_train = rr.predict(X)
            self.w_out_ridge.append(np.transpose(rr.coef_))
            if alpha == 0.01:
                self.w_out = np.transpose(rr.coef_)
                print(" -- Updating w_out using ridge regression , alphe = 0.01-- ")
            # calculate RMSE on train data
            rmse_train = (np.sqrt((Ytrain[1:] - readout_train) ** 2)).mean()
            plt.plot(readout_train,label="alpha: "+str(alpha)+"rmse: "+str(rmse_train))
            print "alpa: "+str(alpha)+", mse_train: "+str(rmse_train)+", w_out_ridge:"
            print self.w_out_ridge[-1]
        plt.legend()
        plt.show()

class Plotting(object):
    def __init__(self, X):
        'Init PCA class'

        # Compute PCA
        self.X = X
        print X.shape

        if self.X.shape[0] > self.X.shape[1]:
            res = matplotlib.mlab.PCA(self.X)
            self.x = res.Y[:, 0]
            self.y = res.Y[:, 1]
        else:
            self.x = self.X[:, 0]
            self.y = self.X[:, 1]

        # Get stats
        self.n = int(self.x.shape[0])

    def __refresh(self, val):
        "Refresh plot"

        plt.plot(self.x[0:val], self.y[0:val], "r-")
        plt.show()
        return "[ " + str(self.x[val]) + " ; " + str(self.y[val]) + " ]"

    def plot_int(self):
        "Plot the 2 first Principal Component in a 2D graph interactively"

        plt.figure()
        # interact(self.__refresh, val=(0, self.n-1, 1))

    def plot_3d(self, n_point=0, end=True):
        "Plot the 2 first Principal Component in a 3D graph"

        if n_point < 2:
            n_point = self.n
        if n_point >= self.x.shape[0]:
            n_point = self.x.shape[0]-1
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        j = 0
        if end:
            r = range(self.n - n_point, self.n - 1)
        else:
            r = range(0, n_point)
        for i in r:
            ax.plot(range(i, i + 2), self.x[i:i + 2], self.y[i:i + 2], color=plt.cm.jet(255 * j / n_point))
            j += 1

        ax.set_xlabel('Epoch')
        ax.set_ylabel('PC 1')
        ax.set_zlabel('PC 2')

    def plot_3d_plotly(self):
        "Plot the 2 first Principal Component in a 3D graph using plotly"

        trace = graph_objs.Scatter3d(
            x=self.x[self.n - 100:self.n], y=self.y[self.n - 100:self.n], z=range(self.n - 100, self.n),
            mode='lines',
            line=dict(
                color=range(0, self.n),
                colorscale='Viridis',
                width=2
            )
        )

        data = [trace]

        layout = dict(
            width=800,
            height=700,
            autosize=False,
            title='First two principal components',
            scene=dict(
                xaxis=dict(
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                ),
                yaxis=dict(
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                ),
                zaxis=dict(
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                ),
                camera=dict(
                    up=dict(
                        x=0,
                        y=0,
                        z=1
                    ),
                    eye=dict(
                        x=-1.7428,
                        y=1.0707,
                        z=0.7100,
                    )
                ),
                aspectratio=dict(x=1, y=1, z=0.7),
                aspectmode='manual'
            ),
        )

        graph = dict(data=data, layout=layout)
        plotly.iplot(graph, filename='RC graph', height=700, validate=False)

        return