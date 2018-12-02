
# coding: utf-8
import matplotlib.pyplot as plt
import matplotlib
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.style.use('fivethirtyeight')
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
from pylab import rcParams

class get(object):

    def figsize(x):
        """
        Default Figsize 8,6
        """
        rcParams['figure.figsize'] = x[0], x[1]

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

    def acf(y):
        return plot_acf(y, ax = plt.gca())

    def pacf(y):
        return plot_pacf(y, ax = plt.gca())

def init(s):
    d = range(0,2)
    p = q = range(0,5)
    P = Q = D = range(0,2)
        
    pdq = list(itertools.product(p,d,q))
    seasonal_pdq_x = list(itertools.product(P,D,Q))
    seasonal_pdq = [(x[0], x[1], x[2], s) for x in seasonal_pdq_x]
    
#     p = d = q = range(0, 2)
#     pdq = list(itertools.product(p, d, q))
#     seasonal_pdq = [(x[0], x[1], x[2], s) for x in pdq]    
    
    return pdq, seasonal_pdq

def calc(ts, s, n):
    
    pdq, seasonal_pdq = init(s)
    
    results_table = bestAIC(ts,pdq,seasonal_pdq)
    
    result, result_summary = Fit(y,results_table)
    
#     plot_prediksi, prediksi = predict(ts, result)
    
#     MSE, RMSE = error(prediksi)
    
#     forecast = Forecast(ts, result, n)
    
#     return result_summary, plot_prediksi, MSE, RMSE, forecast
    return result_summary, result

def bestAIC(ts, pdq, seasonal_pdq):
    results_aic = []
    best_aic = float("inf")
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(ts,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()
#                 print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
            if results.aic < best_aic:
                best_aic = results.aic
            results_aic.append([param,param_seasonal, results.aic]) 
    result_table = pd.DataFrame(results_aic)
    result_table.columns = ['parameters','seasonal_param', 'aic']
#   sorting in ascending order, the lower AIC is - the better
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
    return result_table
    
def Fit(ts, results_table):
    p, d, q = results_table.parameters[0]
    P, D, Q, s = results_table.seasonal_param[0]

    mod = sm.tsa.statespace.SARIMAX(ts,
                                    order=(p, d, q),
                                    seasonal_order=(P, D, Q, s))
    results = mod.fit()
    return results, results.summary()

def predict(ts, result):
    pred = result.get_prediction(start = 0, end = ts.shape[0])
    pred_ci = pred.conf_int()
    ax = y['2014':].plot(label='observed')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
#     plt.legend()
#     return plt.show, pred
    return pred

def error(pred):
    y_forecasted = pred.predicted_mean
    y_truth = y['2017-01-01':]
    mse = ((y_forecasted - y_truth) ** 2).mean()
    rmse = np.sqrt(mse)
    return round(mse, 2), round(rmse, 2)

def Forecast(ts, result, n):
    pred_uc = result.get_forecast(steps=n)
    pred_ci = pred_uc.conf_int()
    ax = ts.plot(label='observed', figsize=(14, 7))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    # plt.legend()
    # return plt.show()