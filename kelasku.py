
# coding: utf-8
import matplotlib.pyplot as plt
import matplotlib
import statsmodels.api as sm
plt.style.use('fivethirtyeight')
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
from pylab import rcParams
# rcParams['figure.figsize'] = 18, 8
class get(object):

    def figsize(x):
        """
        Default Figsize 8,6
        """
        rcParams['figure.figsize'] = x[0], x[1]

    # def figsize(x,y):
    #     self.x = x
    #     self.y = y
    #     return rcParams['figure.figsize'] = x, y

    def trend(s):
        decomposition = sm.tsa.seasonal_decompose(s, model='additive')
        return decomposition.trend.plot()
    
    def seasonal(s):
        decomposition = sm.tsa.seasonal_decompose(s, model='additive')
        return decomposition.seasonal.plot()
    
    def decompose(s):
        decomposition = sm.tsa.seasonal_decompose(s, model='additive')
        decomposition.plot()
        return plt.show()
