import numpy as np
from matplotlib import pyplot as plt
import mxnet as mx
from mxnet import nd
import time

class RingAttractor():

    def __init__(self):

        # parameters of simulation
        self.T = 0.5
        self.Ti = 0.001
        self.dt = 1e-4
        self.Nt = int(np.floor(self.T/self.dt))
        self.Nti = int(np.floor(self.Ti/self.dt))
        self.n = 100
        self.c = nd.zeros((self.n, self.Nt), ctx=mx.gpu(),dtype=np.float32)  # Initialize as an NDArray on the GPU

        self.u = nd.zeros((1, self.Nt), ctx=mx.gpu(),dtype=np.float32)  # Initialize as an NDArray on the GPU
        self.cues = nd.zeros((8, self.n, self.Nt), ctx=mx.gpu(),dtype=np.float32)
        self.c[:, 0] = 0.05 * nd.ones(self.n, ctx=mx.gpu(),dtype=np.float32)  # Initialize 
        print(self.c.shape)


    def generate_signal(self, K, miu, sigma, cue_list, index, noise=0.0):
        Nn = self.n
        Dir = mx.nd.linspace(0, 360 - 360 / Nn, Nn).reshape(Nn, 1).as_in_context(mx.gpu())
        x = nd.zeros((Nn, int(self.Nt)), ctx=mx.gpu())

        diff = mx.nd.minimum(mx.nd.abs(Dir - miu), 360 - mx.nd.abs(Dir - miu))
        c1 = K * mx.nd.exp(-diff**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
        x[:, int(self.Nti):] = mx.nd.repeat(c1.reshape((Nn, 1)), int(self.Nt - self.Nti), axis=1)
        x = x + noise * mx.nd.random.randn(Nn, int(self.Nt), ctx=mx.gpu())
        self.cues[index] = x


    def cue_integration(self, C):
        
        Nn = self.n
        Dir = mx.nd.linspace(0, 360 - 360 / Nn, Nn).reshape(Nn, 1).as_in_context(mx.gpu())
        wEEk = 45.0 / Nn
        sigma = 120.0
        wEE = nd.zeros((Nn, Nn), ctx=mx.gpu(),dtype=np.float32)

        diff = mx.nd.minimum(mx.nd.abs(Dir - Dir.T), 360 - mx.nd.abs(Dir - Dir.T))
        wEE = mx.nd.exp(-diff**2 / (2 * sigma**2)) 

        wEE = wEE * wEEk
        wIE = 60.0 / Nn
        wEI = -6.0
        wII = -1.0
        gammaE = -1.5
        gammaI = -7.5
        tauE = 0.005
        tauI = 0.00025

        for i in range(8):
            self.generate_signal(C[i][0], C[i][1], C[i][2], self.cues, i, noise=0.0)

        start_time = time.time()
        
        for t in range(1, int(self.Nt)):
            
            self.c[:, t] = self.c[:, t - 1] + (-self.c[:, t - 1] + mx.nd.maximum(
                nd.zeros(((Nn,)), ctx=mx.gpu(),dtype=np.float32), gammaE + nd.dot(wEE, self.c[:, t - 1]) + wEI * self.u[:, t - 1] 
                +self.cues [0][:, t - 1]
                +self.cues [1][:, t - 1]
                +self.cues [2][:, t - 1]
                +self.cues [3][:, t - 1]
                +self.cues [4][:, t - 1]
                +self.cues [5][:, t - 1]
                +self.cues [6][:, t - 1]
                +self.cues [7][:, t - 1]
                )) * self.dt / tauE

            
            self.u[:, t] = self.u[:, t - 1] + (-self.u[:, t - 1] + np.max([0, gammaI + wIE * np.sum(self.c[:, t - 1]) + wII * self.u[:, t - 1]],
                                                        axis=0)) * self.dt / tauI
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Elapsed time: ", elapsed_time) 
        return self.c[:, -1].asnumpy()


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
