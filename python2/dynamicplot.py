from __future__ import absolute_import
import matplotlib.pyplot as plt
from itertools import izip

class DynamicPlot(object):
    def __init__(self, title, xdata, ydata):
        if len(xdata) == 0:
            return
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(title, color=u'C0')
        self.ax.set_xlim(xdata[0], xdata[-1])
        self.yline = {}
        for label, data, idx in izip(ydata.keys(), ydata.values(), xrange(len(ydata))):
            if len(xdata) != len(data):
                print u'DynamicPlot::Error: Dimensions of x- and y-data not the same (skipping).'
                continue
            self.yline[label], = self.ax.plot(xdata, data, u'C{}'.format((idx+1)%9), label=u" ".join(label.split(u'_')))
        self.ax.legend()

    def setxlim(self, xliml, xlimh):
        self.ax.set_xlim(xliml, xlimh)

    def setylim(self, yliml, ylimh):
        self.ax.set_ylim(yliml, ylimh)

    def update_plots(self, ydata):
        for k, v in ydata.items():
            self.yline[k].set_ydata(v)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_plot(self, label, data):
        self.yline[label].set_ydata(data)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()