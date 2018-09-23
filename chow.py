import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import scipy
from sklearn.linear_model import LinearRegression

def split_at_breakpoint(ts, breakpoint):
      
    ts1, ts2 = ts[ts.index < breakpoint ], ts[ts.index >= breakpoint ]
    
    return ts.copy(), ts1.copy(), ts2.copy()    

def rss (ts):
    """Fits a linear regression model and computes the residual sum of squares.
    Args:
        ts: The whole time series
    Returns:
        rss: residual sum of squares
    """
    
    x = np.arange(len(ts)).reshape(-1,1)
    y = ts.values
    
    lm = LinearRegression().fit(x, y)
    y_pred = lm.predict(x)
    
    rss = np.sum((y - y_pred)**2)

    return rss

def f_value(ts, ts1, ts2):
    """Computes the f-value of the Chow test.
    Args:
        ts: The whole time series
        ts1: Time series before breakpoint
        ts2: Time series after breakpoint
    Returns:
        F: F-value of Chow test
    """

    rss_total = rss(ts)
    rss_1 = rss(ts1)
    rss_2 = rss(ts2)

    F = ((rss_total - (rss_1 + rss_2)) / 2) / ((rss_1 + rss_2) / (len(ts) - 4))
    
    return F

def chow_test(ts, breakpoint):
    """Performs the Chow test for the given time series at a specified breakpoint.
    Args:
        ts: time series
        breakpoint: date
    Returns:
        p: p-value of the Chow test
    """
    
    ts, ts1, ts2 = split_at_breakpoint(ts, breakpoint)
    
    F = f_value(ts, ts1, ts2)
    
    p = scipy.stats.f.sf(F, 2, len(ts)-4)
    
    return p

def chow_test_all(df, breakpoint):
    """Performs the Chow test for all groups at a specified breakpoint.
    Args:
        df: DataFrame with an id, date and value column
        breakpoint: date
    Returns:
        p_values: p-value of the Chow test for each group given by id
    """
    p_values = df.groupby(level=0).apply(lambda x: pd.Series({'p-value': chow_test(x.reset_index(level=[0], drop=True), breakpoint)}))
    return p_values

def plot_chow(ts, breakpoint, title='', x_label='', y_label=''):
    """Plots the linear regression models computed for the chow test."""
    
    sb.set(color_codes=True)
    pal = sb.color_palette('colorblind')
    
    print('p-value: ' + str(chow_test(ts, breakpoint)))
    
    ts, ts1, ts2 = split_at_breakpoint(ts, breakpoint)
    
    loc = range(0, len(ts)+1, 12)
    labels = range(ts.index.min().year, ts.index.max().year +2)
    
    ts['X'] =np.arange(len(ts))
    ts1['X'] =np.arange(len(ts1))
    ts2['X'] =np.arange(len(ts2)) + len(ts1)

    ts['id'] = '0'
    ts1['id'] = '1'
    ts2['id'] = '2'

    all_ts = pd.concat([ts, ts1, ts2])

    sb.lmplot(data=all_ts, x='X' , y=all_ts.columns[0], hue='id', scatter_kws={"s": 0}, truncate=True, legend=False, ci=None, 
              palette=[pal[4], pal[2], pal[1]])
    plt.plot(ts.X, ts.iloc[:, 0])
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(loc, labels)
    
    
# def prepare_df(df):
#    '''Aggregates the time series in each group.'''
#    return df.groupby([df.columns[0], pd.Grouper(key=df.columns[1], freq='M')]).mean()

# def select_ts(df, identifier):
#     ts = df.loc[df.iloc[:, 0] == identifier, df.columns[[1, 2]]]
#     ts.set_index(df.columns[1], inplace=True)
#     ts.name = identifier
#     return ts