import numpy as np
from matplotlib import pyplot as plt
import time

class RingAttractor():

    def __init__(self):

        # parameters of simulation
        self.T = 0.5
        self.Ti = 0.001
        self.dt = 1e-4
        self.Nt = np.floor(self.T/self.dt)
        self.Nti = np.floor(self.Ti/self.dt)
        self.n = 100
        self.c = np.zeros((self.n, int(self.Nt)))
        self.u = np.zeros((1, int(self.Nt)))
        self.c[:, 0] = 0.05 * np.ones(self.n, )


    def generate_signal(self, K, miu, sigma, noise=0.0):

        Nn = self.n
        Dir = np.linspace(0, 360 - 360 / Nn, Nn).reshape(Nn, 1)
        x = np.zeros((Nn, int(self.Nt)))
        diff = np.min([np.abs(Dir - miu), 360 - np.abs(Dir - miu)], axis=0)
        c1 = K * np.exp(-diff ** 2 / (2 * sigma** 2)) / (np.sqrt(2 * np.pi) * sigma)
        x[:, int(self.Nti):] = np.repeat(c1.reshape(Nn, 1), int(self.Nt - self.Nti), axis=1)
        
        x = x + noise * np.random.randn(Nn, int(self.Nt))

        return x


    def cue_integration(self, C):

        Nn = self.n
        Dir = np.linspace(0, 360 - 360 / Nn, Nn).reshape(Nn, 1)

        # parameters of ring attractor network
        wEEk = 45.0 / Nn
        wEE = np.zeros((Nn, Nn))
        sigma = 120.0

        # for i in range(0, Nn):
        #     for j in range(0, Nn):
        #         diff = np.min([np.abs(Dir[i] - Dir[j]), 360 - np.abs(Dir[i] - Dir[j])])

        #         wEE[i, j] = np.exp((-diff ** 2) / (2 * sigma ** 2))

        diff_matrix = np.minimum(np.abs(Dir - Dir.T), 360 - np.abs(Dir - Dir.T))
        wEE = np.exp(-diff_matrix**2 / (2 * sigma**2))

        wEE = wEE * wEEk
        wIE = 60.0 / Nn

        wEI = -6.0
        wII = -1.0

        gammaE = -1.5
        gammaI = -7.5

        tauE = 0.005
        tauI = 0.00025
        
        x0 = self.generate_signal(C[0][0], C[0][1], C[0][2], noise=0.0)
        x1 = self.generate_signal(C[1][0], C[1][1], C[1][2], noise=0.0)
        x2 = self.generate_signal(C[2][0], C[2][1], C[2][2], noise=0.0)
        x3 = self.generate_signal(C[3][0], C[3][1], C[3][2], noise=0.0)
        x4 = self.generate_signal(C[4][0], C[4][1], C[4][2], noise=0.0)
        x5 = self.generate_signal(C[5][0], C[5][1], C[5][2], noise=0.0)
        x6 = self.generate_signal(C[6][0], C[6][1], C[6][2], noise=0.0)
        x7 = self.generate_signal(C[7][0], C[7][1], C[7][2], noise=0.0)

        # run integration
        for t in range(1, int(self.Nt)):
            self.c[:, t] = self.c[:, t - 1] + (-self.c[:, t - 1] + np.max(
                [np.zeros((Nn,)), gammaE + np.dot(wEE, self.c[:, t - 1]) + wEI * self.u[:, t - 1] 
                + x0[:, t - 1]
                + x1[:, t - 1]
                + x2[:, t - 1]
                + x3[:, t - 1]
                + x4[:, t - 1]
                + x5[:, t - 1]
                + x6[:, t - 1]
                + x7[:, t - 1]
                ], axis=0)) * self.dt / tauE
            

            self.u[:, t] = self.u[:, t - 1] + (-self.u[:, t - 1] + np.max([0, gammaI + wIE * np.sum(self.c[:, t - 1]) + wII * self.u[:, t - 1]],
                                                        axis=0)) * self.dt / tauI

        return self.c[:, -1]


def main(args=None):
    
    RA = RingAttractor()

    # parameter Path Integration
    miu_PI = 90
    k_PIs = [40, 20, 5]
    sigma_PI = 40

    # parameter Vision
    miu_V = 160
    k_V = 40
    sigma_V = 40
    n = 100

    z = RA.cue_integration([[40,160,40], [30,12,40], [40,90,40], [40,0.1,40], [40,90,40], [40,90,40], [40,90,40], [40,90,40]])

    print(np.argmax(z)*360/100)

    # z = RA.cue_integration([[40,160,40], [20,90,40]])

    # print(np.argmax(z)*360/100)
    # z = RA.cue_integration([[40,160,40], [5,90,40]])

    # print(np.argmax(z)*360/100)



    plt.show()


if __name__ == '__main__':
	main()
