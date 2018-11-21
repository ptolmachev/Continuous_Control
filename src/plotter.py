import pickle
from scipy.signal import savgol_filter
import numpy as np
from matplotlib import pyplot as plt
def plotter(scores, threshold):

    #PLOT SCORES
    x = np.arange(len(scores))
    xs = threshold*np.ones(len(scores))
    plt.xkcd()
    plt.plot(xs,'k--', linewidth = 2)
    plt.plot(savgol_filter(scores,11,3), linewidth = 3, color = 'Red')

    plt.title("Gathered reward per episode")
    plt.xlabel("Number of an episode")
    plt.ylabel("Score")
    plt.grid(True)
    plt.show()

# scores = pickle.load(open('../results/Reacher.pkl', 'rb+'))
# plotter(scores, 30)