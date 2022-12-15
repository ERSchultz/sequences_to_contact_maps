'''Taken from Soren: https://github.com/SorenKyhl/TICG-chromatin/blob/merge/pylib/chipseqPipeline.py'''
import matplotlib.pyplot as plt
import numpy as np
import scipy
from pylib import epilib


class ChipseqPipeline():
    def __init__(self, operations, plot = False):
        self.operations = operations
        self.plot = plot

    def fit(self, x):
        for operation in self.operations:
            x = operation.operate(x, self.plot)
        return x

class Smooth:
    def __init__(self, size=2):
        self.size = size

    def operate(self, x, plot):
        return scipy.ndimage.gaussian_filter(x, self.size)

class Normalize:
    def __init__(self):
        pass

    def new_map_0_1_chip(self, signal, plot):
        baseline = self.get_baseline_signal(signal, plot)
        signal = (signal - baseline)/(np.max(signal) - baseline)
        signal[signal<0] = 0
        signal *= 1.5
        signal[signal>1] = 1
        return signal

    def get_baseline_signal(self, signal, plot):
        bin_populations, bin_signal = np.histogram(signal, bins=100)
        baseline_signal = bin_signal[np.argmax(bin_populations[1:])+1]
        if plot:
            plt.hist(signal, bins = 100)
            plt.ylabel('Count')
            plt.xlabel('ChIP-seq Value')
            plt.axvline(baseline_signal, color = 'k', label = f'baseline_signal = {baseline_signal}')
            plt.legend()
            plt.show()
            plt.close()
        return baseline_signal

    def operate(self, x, plot):
        # parameter: scaling not exposed
        normalized = self.new_map_0_1_chip(x, plot)
        if plot:
            plt.plot(x, label = 'input')
            plt.plot(normalized, label = 'normalized')
            plt.legend()
            plt.show()
            plt.close()
        centered = 2*normalized - 1
        return centered

class Sigmoid:
    def __init__(self, w=20, b=10):
        self.w = w
        self.b = b

    def operate(self, x, plot):
        # perhaps this should just be a tanh?
        def sigmoid_fn(x):
            return 1 / (1 + np.exp(-x))

        x_out = sigmoid_fn(self.w*x+self.b)
        x_out = 2*x_out-1

        if plot:
            plt.plot(x, label = 'input')
            plt.plot(x_out, label = 'sigmoid')
            plt.legend()
            plt.show()
            plt.close()

        return x_out
