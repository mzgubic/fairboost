import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_toys(n_samples, z=None, train=True, pandas=False):
    """
    Args:
        n_samples (int): number of examples to generate
        z (None or float): generate events at a particular value of z, by default depends on the label
        
    Returns a tuple (X, Y, Z) of arrays. In each array, a row is an instance.
    """
    
    sigma = 1.0

    # labels Y: first half are zeros, second half are ones
    Y = np.zeros(n_samples)
    Y[n_samples//2:] = 1

    # protected parameter Z
    # in the training set (z=None) z depends on the label
    # (i.e. mass distribution depends on signal/background)
    if z == None:
        Z0 = np.zeros(n_samples//2)
        Z1 = np.random.normal(0, 1, size=n_samples//2)
        #Z1 = np.random.uniform(-1, 2, size=n_samples//2)
    # or generate at specific value of Z (for evaluation purposes)
    else:
        Z0 = z * np.ones(n_samples//2)
        Z1 = z * np.ones(n_samples//2)
    Z = np.concatenate([Z0, Z1])

    # feature X depend on the value of Z (i.e. kinematics depends on the mass)
    X0 = np.random.multivariate_normal([0, 0], [[sigma, -.5*sigma], [-.5*sigma, sigma]], size=n_samples//2)
    X1 = np.random.multivariate_normal([1, 1], 0.5*np.eye(2), size=n_samples//2)
    X1[:,1] += Z1
    X = np.concatenate([X0, X1])

    if not pandas:
        return X, Y, Z
    if pandas:
        XZ = np.concatenate([X, Z.reshape(-1,1)], axis=1)
        X_df = pd.DataFrame(XZ, columns=['x1', 'x2', 'z'])
        return X_df, Y, Z


def show_variates(ax, generate):
    """
    Plots the random variates.
    
    Args:
        ax (matplotlib axis): Axis on which to plot the variates.
        generate (function): function to generate the variates.
    """
    
    # generate
    n_samples = 5000
    X0, Y0, Z0 = generate(n_samples, z=0)
    Y0 = Y0.ravel()
    Z0 = Z0.ravel()
    X1, Y1, Z1 = generate(n_samples, z=1)
    Y1 = Y1.ravel()
    Z1 = Z1.ravel()
    X_1, Y_1, Z_1 = generate(n_samples, z=-1)
    Y_1 = Y_1.ravel()
    Z_1 = Z_1.ravel()
    
    # plot
    ax.scatter(X0[Y0==0,0], X0[Y0==0,1], marker='o', color='k', alpha=1.0, label='Y=0')
    ax.scatter(X_1[Y_1==1,0], X_1[Y_1==1,1], marker='x', c='tomato', alpha=0.3, label='Y=1, Z=-1')
    ax.scatter(X0[Y0==1,0], X0[Y0==1,1], marker='x', c='red', alpha=0.3, label='Y=1, Z=0')
    ax.scatter(X1[Y1==1,0], X1[Y1==1,1], marker='x', c='darkred', alpha=0.3, label='Y=1, Z=1')
    
    
    # cosmetics
    ax.set_ylim(-1, 3)
    ax.set_xlim(-1, 2)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    leg = ax.legend(loc='best')
    ax.set_title('Variates')

