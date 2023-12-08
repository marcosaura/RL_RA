import numpy as np
from matplotlib import pyplot as plt


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
        print("cues :", x)
        return x


    def cue_integration(self, C1, C2):
        # ring attractor function
        Nn = self.n
        Dir = np.linspace(0, 360 - 360 / Nn, Nn).reshape(Nn, 1)

        # parameters of ring attractor network
        wEEk = 45.0 / Nn
        wEE = np.zeros((Nn, Nn))
        sigma = 120.0
        for i in range(0, Nn):
            for j in range(0, Nn):
                diff = np.min([np.abs(Dir[i] - Dir[j]), 360 - np.abs(Dir[i] - Dir[j])])
                wEE[i, j] = np.exp((-diff ** 2) / (2 * sigma ** 2))

        wEE = wEE * wEEk
        wIE = 60.0 / Nn

        wEI = -6.0
        wII = -1.0

        gammaE = -1.5
        gammaI = -7.5

        tauE = 0.005
        tauI = 0.00025

        x1 = self.generate_signal(C1[0], C1[1], C1[2], noise=0.0)
        x2 = self.generate_signal(C2[0], C2[1], C2[2], noise=0.0)

        # run integration

        for t in range(1, int(self.Nt)):
            self.c[:, t] = self.c[:, t - 1] + (-self.c[:, t - 1] + np.max(
                [np.zeros((Nn,)), gammaE + np.dot(wEE, self.c[:, t - 1]) + wEI * self.u[:, t - 1] 
                + x1[:, t - 1]
                + x2[:, t - 1]
                ], axis=0)) * self.dt / tauE
            print(self.c[:, t])

            self.u[:, t] = self.u[:, t - 1] + (-self.u[:, t - 1] + np.max([0, gammaI + wIE * np.sum(self.c[:, t - 1]) + wII * self.u[:, t - 1]],
                                                        axis=0)) * self.dt / tauI
            
            print( self.u[:, t])
        return x1[:, -1], x2[:, -1], self.c[:, -1]


    def plotter(self, n,pi,v,c,k_PI):
        Nn = n
        Dir = np.linspace(0, 360 - 360 / Nn, Nn).reshape(Nn, 1)
        fontsize_k = 30
        fig = plt.figure(figsize=(19.2, 10.8))
        plt.plot(Dir, pi, color='red', lw=2.0, label='PI Signal')
        plt.plot(Dir, v, color='black', lw=2.0, label='Vision Signal')
        plt.plot(Dir, c, color='green', lw=2.0, label='Ring Attractor Integration')
        # plt.plot(Dir, opt_it1, 'b--', lw=2.0, label='MLE')
        plt.title(r'Activation Profile for PI Descending to Zero($N=100, \xi=0.0$)',
                fontdict={'size': fontsize_k, 'color': 'k'})
        plt.text(0, 0.6, r'$K_{Vision}=40$', fontdict={'size': fontsize_k - 4, 'color': 'k'})
        plt.text(0, 0.55, r'$K_{PI}=$' + str(k_PI), fontdict={'size': fontsize_k - 4, 'color': 'r'})
        plt.legend(fontsize=fontsize_k - 4)
        plt.ylabel('Activation', fontdict={'size': fontsize_k, 'color': 'k'})
        xticks = np.linspace(0, 360, Nn)
        plt.xticks(xticks, visible=0)
        ymax = np.max(c)
        plt.ylim(-0.05, ymax + 0.05)
        # plt.xlabel('Neurons Labelled with Preferences', fontdict={'size': fontsize_k, 'color': 'k'})
        plt.grid(1)
        return fig


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

    x, y, z = RA.cue_integration([40,160,40], [40,90,40])
    sub_fig1 = RA.plotter(n,x,y,z,k_PIs[0])

    print(np.argmax(z)*360/)

    x, y, z = RA.cue_integration([40,160,40], [20,90,40])
    sub_fig2 = RA.plotter(n,x,y,z,k_PIs[1])


    print(np.argmax(z))
    x, y, z = RA.cue_integration([40,160,40], [5,90,40])
    sub_fig3 = RA.plotter(n,x,y,z,k_PIs[2])

    print(np.argmax(z))



    plt.show()


if __name__ == '__main__':
	main()
