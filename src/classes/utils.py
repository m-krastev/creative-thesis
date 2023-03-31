from typing import Sequence
import numpy.typing as ntp
import numpy as np

def mean(items: Sequence) -> float:
    return sum(items)/len(items)

def cross_softmax(x: ntp.NDArray, y: ntp.NDArray, temp1=0.5, temp2=0.5):
    # return (torch.softmax(torch.from_numpy(results[1][:,0]), dim=0) @ torch.softmax(torch.from_numpy(results[1][:,1]), dim=0)).item()
    exps = np.exp(x*temp1)
    exps /= exps.sum()
    exps2 = np.exp(y*temp2)
    exps2 /= exps2.sum()
    return exps @ exps2


def slope_coefficient(X: ntp.NDArray, Y: ntp.NDArray) -> float:
    """Returns the coefficient of the slope"""
    # Using the integrated function
    # return np.tanh(np.polyfit(X,Y,1)[0])
    # Manually implementing slope equation
    return ((X*Y).mean(axis=0) - X.mean()*Y.mean(axis=0)) / ((X**2).mean() - (X.mean())**2)
