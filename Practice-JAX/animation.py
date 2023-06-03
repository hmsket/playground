# JAX関連のディレクトリで、
# matplotlib.animationの練習をします！

# https://python-academia.com/matplotlib-animation/

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_hist(data):
    plt.cla()
    x = np.random.normal(0, 10, 1000)
    frame = plt.hist(x, bins=20, range=(-50,50), density=True, ec='black')


fig = plt.figure(figsize=(5,5), facecolor='lightblue')

ani = FuncAnimation(fig, plot_hist, interval=10, frames=10)

ani.save('./hist.gif', writer='pillow')
