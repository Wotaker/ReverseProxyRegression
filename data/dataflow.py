import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def np_to_csv(array: np.ndarray, path: str) -> None:
    pd.DataFrame(array).to_csv(path, index=False)
    return

def np_read_csv(path: str) -> np.ndarray:
    return np.array(pd.read_csv(path))

def np_serialize(array: np.ndarray) -> tuple[bytes, tuple]:
    return array.astype(np.float64).tobytes(), array.shape

def np_deserialize(array_bytes: bytes, array_shape: tuple) -> np.ndarray:
    return np.reshape(np.frombuffer(array_bytes), array_shape)


# Plot a linear regression fit
#   ax - axis for plotting,
#   w - linear regression parameters
#   data - explanatory variables and responses
#   test_x, test_y, test_ys - explanatory variables, predictions and their uncertainties (test)
#   RSS - residual  sum of squares
#   title - plot title.
def plot_fit(ax, w, data, data_ys, 
             test_x=None, test_y=None, test_ys=None,
             RSS=None, title=None):
    
    data_x, data_y = data[:, 0], data[:, 1]
    data_x, data_y = np.reshape(data_x, (-1, 1)), np.reshape(data_y, (-1, 1))
    ones = np.ones((data_x.shape[0], 1))
    data_x = np.concatenate((data_x, ones), axis=1)

    xmin, xmax = np.min(data_x[:, 0]), np.max(data_x[:, 0])
    if test_x is not None:
        test_min, test_max = np.min(test_x), np.max(test_x)
        xmin = min(xmin, test_min)
        xmax = max(xmax, test_max)
    
    X = np.array([[xmin, 1], [xmax, 1]])
    y = X @ w
    
    ax.errorbar(data_x[:, 0], data_y[:, 0], data_ys, None, marker="o", ls='', capsize=5)
    if test_x is not None:
        ax.errorbar(test_x, test_y, test_ys, None, marker="x", c='g', ls='', capsize=5);
    
    ax.plot(X[:, 0], y, marker='', lw=2.0, color='r');
    
    ax.set_xlabel('x', fontsize='xx-large')
    ax.set_ylabel('y', fontsize='xx-large')
    if RSS is not None:
        ax.text(0.95, 0.01, 'Residual sum of squares: {0:0.1f}'.format(RSS),
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes, fontsize=15)
    
    if title is not None:
        ax.set_title(title)


# Plot samples from a Bayesian linear regression (as a set of lines)
#   ax - axis for plotting,
#   w_samples - samples from the posterior over regression parameters
#   data_x, data_y, data_ys - explanatory variables, responses and their uncertainties (data)
#   title - plot title.
def plot_posterior_samples(ax, w_samples, data_x, data_y, data_ys, title=None):
    ax.errorbar(data_x[:, 0], data_y, data_ys, None, marker="o", ls='', capsize=5)
    ax.set_xlabel('x', fontsize='xx-large')
    ax.set_ylabel('y', fontsize='xx-large')

    xmin, xmax = np.min(data_x[:, 0]), np.max(data_x[:, 0])
    X = np.array([[xmin, 1], [xmax, 1]])
    
    for w in w_samples:
        y = X @ w
        ax.plot(X[:, 0], y, marker='', lw=1.0, alpha=0.5, color='r');
    
    if title is not None:
        ax.set_title(title)


def plot_gpr(ax, V, mu, Sigma, U, y, title=None):
    '''
    Plot Gaussian Process Regression results.
    
    Args:
        ax:    Axis for plotting.
        V:     Regression points.
               shape m times 1.
        mu:    Expected value of f(V),
               shape: m times 1.
        Sigma: Covariance of f(V),
               shape: m times m.
        U:     Points at which values of f were observed,
               shape: n times 1.
        f_U:   True values of f(U),
               shape: n times 1.
        y:     Noisy observations of f(U),
               shape: n times 1.
        sigma: Assumed noise level (standard deviation of y).
        title: Plot title.
    
    Note:
        Normally, f(U) is not know (observations are noisy). We use it
        only to illustrate regression results.
    '''
    ax.plot(V, mu, marker='', lw=2.0, color='r')
    
    f_V_2sigma = 2 * np.sqrt(np.diag(Sigma))
    
    V, mu = V.squeeze(), mu.squeeze()
    plt.fill_between(V, mu + f_V_2sigma, mu - f_V_2sigma, alpha=0.3)
    
    ax.scatter(U, y, s=100, c='g', marker='x', zorder=2)
    # ax.errorbar(U, f_U[:, 0], sigma, None, marker="o", ls='', capsize=5)

    ax.set_xlabel(r'$x$', fontsize='xx-large')
    ax.set_ylabel(r'$f(x)$', fontsize='xx-large')
    
    if title is not None:
        ax.set_title(title)
