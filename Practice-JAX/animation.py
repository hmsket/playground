# JAX関連のディレクトリで、
# matplotlib.animationの練習をします！

# https://python-academia.com/matplotlib-animation/

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_hist(idx, array):
    plt.cla()
    x = np.random.normal(0, 10, 25)
    x = np.reshape(x, (5,5))
    frame = plt.imshow(x, cmap=plt.cm.gray_r)


fig = plt.figure(figsize=(5,5), facecolor='lightblue')

ani = FuncAnimation(fig, plot_hist, fargs=[[1,2]], interval=10, frames=10)

ani.save('./hist.gif', writer='pillow')
