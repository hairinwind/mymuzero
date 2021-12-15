import matplotlib.pyplot as plt
import time
import numpy as np

def draw1(): 
    plt.ion()
    plt.figure(1)
    t = [0]
    t_now = 0
    m = [0]

    for i in range(3):
        plt.clf()
        t.append(i)
        m.append(i*i)
        plt.plot(t,m, '-r')
        plt.draw()
        input("press enter") # the input here is to keep the figure alive. plt.draw() cannot keep figure alive

def draw2():
    plt.ion()
    plt.show()
    x = np.arange(0, 51)               # x coordinates  
    for z in range(10, 50):
        y = np.power(x, z/10)          # y coordinates of plot for animation
        plt.cla()                      # delete previous plot
        # plt.axis([-50, 50, 0, 10000])  # set axis limits, to avoid rescaling
        plt.plot(x, y)                 # generate new plot
        # plt.pause(0.1) 
        input("press enter")



if __name__ == "__main__":
    draw2()