import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
#from utils.misc_tools import error_ellipse
from misc_tools import error_ellipse
from matplotlib.patches import Ellipse


def plot_single_graph(X_Y, title, xlabel, ylabel, label, is_scatter=False, sigma=None):
    """
    That function plots a single graph

    Args:
        X_Y (np.ndarray): array of values X and Y, array shape [N, 2]
        title (str): sets figure title
        xlabel (str): sets xlabel value
        ylabel (str): sets ylabel value
        label (str): sets legend's label value
    """
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if is_scatter:
        plt.scatter(np.arange(X_Y.shape[0]), X_Y, s=1, label=label, c="b")
        if sigma is not None:
            plt.plot(np.arange(X_Y.shape[0]), sigma, c="orange")
            plt.plot(np.arange(X_Y.shape[0]), -sigma, c="orange")
    elif len(X_Y.shape) ==1:
        plt.plot(np.arange(X_Y.shape[0]), X_Y, label=label)
    else:
        plt.plot(X_Y[:, 0], X_Y[:, 1], label=label)
    
    plt.legend()


def plot_graph_and_scatter(X_Y0, X_Y1, title, xlabel, ylabel, label0, label1, color0='b', color1='r', point_size=1):
    """
    That function plots two graphs, plot and scatter

    Args:
        X_Y0 (np.ndarray): array of values X and Y, array shape [N, 2] of graph 0
        X_Y1 (np.ndarray): array of values X and Y, array shape [N, 2] of graph 1
        title (str): sets figure title
        xlabel (str): sets xlabel value
        ylabel (str): sets ylabel value
        label0 (str): sets legend's label value of graph 0
        label1 (str): sets legend's label value of graph 1
        color0 (str): color of graph0
        color1 (str): color of graph1
        point_size(float): size of scatter points
    """
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(X_Y0[:, 0], X_Y0[:, 1], label=label0, c=color0)
    plt.scatter(X_Y1[:, 0], X_Y1[:, 1], label=label1, s=point_size, c=color1)
    plt.legend()


def plot_four_graphs(X_values, Y0, Y1, Y2, Y3, title, xlabel, ylabel, label0, label1, label2, label3):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(X_values, Y0, label=label0)
    plt.plot(X_values, Y1, label=label1)
    plt.plot(X_values, Y2, label=label2)
    plt.plot(X_values, Y3, label=label3)
    plt.legend()


def plot_three_graphs(X_Y0, X_Y1, X_Y2, title, xlabel, ylabel, label0, label1, label2, shapeandcolor0 = 'b-', shapeandcolor1 = 'k-', shapeandcolor2 = 'g:'):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(X_Y0[:, 0], X_Y0[:, 1], '{}'.format(shapeandcolor0), label=label0)
    plt.plot(X_Y1[:, 0], X_Y1[:, 1], '{}'.format(shapeandcolor1), label=label1)
    plt.plot(X_Y2[:, 0], X_Y2[:, 1], '{}'.format(shapeandcolor2), label=label2)
    plt.legend()

def plot_two_graphs_one_double(X_Y0, X_Y1, X_Y2  ,title, xlabel, ylabel, label0, label1):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(X_Y0[:, 0], X_Y0[:, 1], 'b-', label=label0)
    plt.plot(X_Y1[:, 0], X_Y1[:, 1], 'y-', label=label1)
    plt.plot(X_Y2[:, 0], X_Y2[:, 1], 'y-')
    plt.legend()


def build_animation(X_Y0, X_Y1, X_Y2, x_xy_xy_y, title, xlabel, ylabel, label0, label1, label2):
    frames = []
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    print("Creating animation")
    
    x0, y0, x1, y1, x2, y2 = [], [], [], [], [], []
    val0, = plt.plot([], [], 'b-', animated=True, label=label0)
    val1, = plt.plot([], [], 'k-', animated=True, label=label1)
    val2, = plt.plot([], [], 'r--', animated=True, label=label2)
    val3 = Ellipse(xy=(0,0), width=0, height=0, angle=0, animated=True)
    
    ax.add_patch(val3)
    plt.legend()
    
    values = np.hstack((X_Y0, X_Y1, X_Y2, x_xy_xy_y))
    
    def init():
        ax.set_xlim(-200, 50)
        ax.set_ylim(-20, 160)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        val0.set_data([],[])
        val1.set_data([],[])
        val2.set_data([],[])
        
        return val0, val1, val2, val3
    
    def update(frame):
        x0.append(frame[0])
        y0.append(frame[1])
        x1.append(frame[2])
        y1.append(frame[3])
        x2.append(frame[4])
        y2.append(frame[5])
        val0.set_data(x0, y0)
        val1.set_data(x1, y1)
        val2.set_data(x2, y2)
        #print("frame.shape " , frame.shape)
        #print("frame " , frame)
        cov_mat = frame[6:].reshape(2,-1)
        ellipse = error_ellipse(np.array([frame[4], frame[5]]), cov_mat)
        
        #print("ellipse.get_angle()" , ellipse.get_angle())
        val3.angle = ellipse.angle
        val3.set_center(ellipse.get_center())
        val3.width = ellipse.width
        val3.height = ellipse.height
        #val3.set_alpha(ellipse.alpha())
        
        return val0, val1, val2, val3
    
    anim = animation.FuncAnimation(fig, update, frames=values, init_func=init, interval=1, blit=True)
    return anim


def save_animation(ani, basedir, file_name):
    print("Saving animation")
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=50, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(os.path.join(basedir, f'{file_name}.mp4'), writer=writer)
    print("Animation saved")


def show_graphs():
    plt.show()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from matplotlib import animation

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    e1 = Ellipse(xy=(0.5, 0.5), width=0.5, height=0.2, angle=60, animated=True)
    e2 = Ellipse(xy=(0.8, 0.8), width=0.5, height=0.2, angle=100, animated=True)
    ax.add_patch(e1)
    ax.add_patch(e2)

    def init():
        return [e1, e2]

    def animate(i):
        e1.angle = e1.angle + 0.5
        e2.angle = e2.angle + 0.5
        return e1, e2

    anim = animation.FuncAnimation(fig, animate, init_func=init, interval=1, blit=True)
    plt.show()