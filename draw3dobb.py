from __future__ import print_function, division
from matplotlib import pyplot as plt
import numpy as np
from numpy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D

def tryPlot():
    cmap = plt.get_cmap('jet_r')
    fig = plt.figure()
    ax = Axes3D(fig)
    draw(ax, [-0.0152730000000000,-0.113074400000000,0.00867852000000000,0.766616000000000,0.483920000000000,0.0964542000000000,
               8.65505000000000e-06,-0.000113369000000000,0.999997000000000,0.989706000000000,0.143116000000000,7.65900000000000e-06], cmap(float(1)/7))
    draw(ax, [-0.310188000000000,0.188456800000000,0.00978854000000000,0.596362000000000,0.577190000000000,0.141414800000000,
               -0.331254000000000,0.943525000000000,0.00456327000000000,-0.00484978000000000,-0.00653891000000000,0.999967000000000], cmap(float(2)/7))
    draw(ax, [-0.290236000000000,-0.334664000000000,-0.328648000000000,0.322898000000000,0.0585966000000000,0.0347996000000000,
               -0.330345000000000,-0.942455000000000,0.0514932000000000,0.0432524000000000,0.0393726000000000,0.998095000000000], cmap(float(3)/7))
    draw(ax, [-0.289462000000000,-0.334842000000000,0.361558000000000,0.322992000000000,0.0593536000000000,0.0350418000000000,
               0.309240000000000,0.949730000000000,0.0485183000000000,-0.0511885000000000,-0.0343219000000000,0.998099000000000], cmap(float(4)/7))
    draw(ax, [0.281430000000000,-0.306584000000000,0.382928000000000,0.392156000000000,0.0409424000000000,0.0348472000000000,
               0.322342000000000,-0.942987000000000,0.0828920000000000,-0.0248683000000000,0.0791002000000000,0.996556000000000], cmap(float(5)/7))
    draw(ax, [0.281024000000000,-0.306678000000000,-0.366110000000000,0.392456000000000,0.0409366000000000,0.0348446000000000,
               -0.322608000000000,0.942964000000000,0.0821142000000000,0.0256742000000000,-0.0780031000000000,0.996622000000000], cmap(float(6)/7))
    draw(ax, [0.121108800000000,-0.0146729400000000,0.00279166000000000,0.681576000000000,0.601756000000000,0.0959706000000000,
               -0.986967000000000,-0.160173000000000,0.0155341000000000,0.0146809000000000,0.00650174000000000,0.999801000000000], cmap(float(7)/7))
    plt.show()

def draw(ax, p, color):
    tmpPoint = p

    center = tmpPoint[0: 3]
    lengths = tmpPoint[3: 6]
    dir_1 = tmpPoint[6: 9]
    dir_2 = tmpPoint[9: ]

    dir_1 = dir_1/LA.norm(dir_1)
    dir_2 = dir_2/LA.norm(dir_2)
    dir_3 = np.cross(dir_1, dir_2)
    dir_3 = dir_3/LA.norm(dir_3)
    cornerpoints = np.zeros([8, 3])

    d1 = 0.5*lengths[0]*dir_1
    d2 = 0.5*lengths[1]*dir_2
    d3 = 0.5*lengths[2]*dir_3

    cornerpoints[0][:] = center - d1 - d2 - d3
    cornerpoints[1][:] = center - d1 + d2 - d3
    cornerpoints[2][:] = center + d1 - d2 - d3
    cornerpoints[3][:] = center + d1 + d2 - d3
    cornerpoints[4][:] = center - d1 - d2 + d3
    cornerpoints[5][:] = center - d1 + d2 + d3
    cornerpoints[6][:] = center + d1 - d2 + d3
    cornerpoints[7][:] = center + d1 + d2 + d3

    ax.plot([cornerpoints[0][0], cornerpoints[1][0]], [cornerpoints[0][1], cornerpoints[1][1]],
            [cornerpoints[0][2], cornerpoints[1][2]], c=color)
    ax.plot([cornerpoints[0][0], cornerpoints[2][0]], [cornerpoints[0][1], cornerpoints[2][1]],
            [cornerpoints[0][2], cornerpoints[2][2]], c=color)
    ax.plot([cornerpoints[1][0], cornerpoints[3][0]], [cornerpoints[1][1], cornerpoints[3][1]],
            [cornerpoints[1][2], cornerpoints[3][2]], c=color)
    ax.plot([cornerpoints[2][0], cornerpoints[3][0]], [cornerpoints[2][1], cornerpoints[3][1]],
            [cornerpoints[2][2], cornerpoints[3][2]], c=color)
    ax.plot([cornerpoints[4][0], cornerpoints[5][0]], [cornerpoints[4][1], cornerpoints[5][1]],
            [cornerpoints[4][2], cornerpoints[5][2]], c=color)
    ax.plot([cornerpoints[4][0], cornerpoints[6][0]], [cornerpoints[4][1], cornerpoints[6][1]],
            [cornerpoints[4][2], cornerpoints[6][2]], c=color)
    ax.plot([cornerpoints[5][0], cornerpoints[7][0]], [cornerpoints[5][1], cornerpoints[7][1]],
            [cornerpoints[5][2], cornerpoints[7][2]], c=color)
    ax.plot([cornerpoints[6][0], cornerpoints[7][0]], [cornerpoints[6][1], cornerpoints[7][1]],
            [cornerpoints[6][2], cornerpoints[7][2]], c=color)
    ax.plot([cornerpoints[0][0], cornerpoints[4][0]], [cornerpoints[0][1], cornerpoints[4][1]],
            [cornerpoints[0][2], cornerpoints[4][2]], c=color)
    ax.plot([cornerpoints[1][0], cornerpoints[5][0]], [cornerpoints[1][1], cornerpoints[5][1]],
            [cornerpoints[1][2], cornerpoints[5][2]], c=color)
    ax.plot([cornerpoints[2][0], cornerpoints[6][0]], [cornerpoints[2][1], cornerpoints[6][1]],
            [cornerpoints[2][2], cornerpoints[6][2]], c=color)
    ax.plot([cornerpoints[3][0], cornerpoints[7][0]], [cornerpoints[3][1], cornerpoints[7][1]],
            [cornerpoints[3][2], cornerpoints[7][2]], c=color)

def showGenshapes(genshapes):
    for i in range(len(genshapes)):
        recover_boxes = genshapes[i]

        fig = plt.figure(i)
        cmap = plt.get_cmap('jet_r')
        ax = Axes3D(fig)
        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.7, 0.7)
        ax.set_zlim(-0.7, 0.7)

        for jj in range(len(recover_boxes)):
            p = recover_boxes[jj][:]
            draw(ax, p, cmap(float(jj)/len(recover_boxes)))

        plt.show()

def showGenshape(genshape):
    recover_boxes = genshape

    fig = plt.figure(0)
    cmap = plt.get_cmap('jet_r')
    ax = Axes3D(fig)
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
    ax.set_zlim(-0.7, 0.7)

    for jj in range(len(recover_boxes)):
        p = recover_boxes[jj][:]
        draw(ax, p, cmap(float(jj)/len(recover_boxes)))

    plt.show()
