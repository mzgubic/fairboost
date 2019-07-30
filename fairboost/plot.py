import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def show_roc_curves(ax, clf, generate, pandas=False):

    # generate test data
    n_samples = 10000
    X0, Y0, Z0 = generate(n_samples, z=0, pandas=pandas)
    X1, Y1, Z1 = generate(n_samples, z=1, pandas=pandas)
    X_1, Y_1, Z_1 = generate(n_samples, z=-1, pandas=pandas)

    # compute the roc curves
    Y0_pred = clf.predict_proba(X0)[:,1]
    fpr0, tpr0, _ = roc_curve(Y0, Y0_pred)

    Y1_pred = clf.predict_proba(X1)[:,1]
    fpr1, tpr1, _ = roc_curve(Y1, Y1_pred)

    Y_1_pred = clf.predict_proba(X_1)[:,1]
    fpr_1, tpr_1, _ = roc_curve(Y_1, Y_1_pred)

    # draw the roc curves
    ax.plot(fpr_1, tpr_1, c='tomato', label='Z=-1')
    ax.plot(fpr0, tpr0, c='red', label='Z=0')
    ax.plot(fpr1, tpr1, c='darkred', label='Z=1')

    # cosmetics
    ax.legend(loc='best')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('ROC curve')

def show_decision_boundary(ax, clf, generate, pandas=False):

    # generate test data
    n_samples = 100000
    X, Y, Z = generate(n_samples, pandas=pandas)

    # predict
    preds = clf.predict_proba(X)[:,1]

    # plot
    if pandas:
        X = X.values
    dec = ax.tricontourf(X[:,0], X[:,1], preds.ravel(), 20, extend='both')
    plt.colorbar(dec, ax=ax)

    # cosmetics
    ax.set_ylim(-1, 3)
    ax.set_xlim(-1, 2)
    ax.set_title('Decision boundary')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')


def show_clf_output(ax, clf, generate, pandas=False):

    # generate test data
    n_samples = 10000
    X0, Y0, Z0 = generate(n_samples, z=0, pandas=pandas)
    X1, Y1, Z1 = generate(n_samples, z=1, pandas=pandas)
    X_1, Y_1, Z_1 = generate(n_samples, z=-1, pandas=pandas)

    # compute the predictions
    Y0_pred = clf.predict_proba(X0)[:,1]
    Y1_pred = clf.predict_proba(X1)[:,1]
    Y_1_pred = clf.predict_proba(X_1)[:,1]

    # show them
    bins = 30
    ax.hist(Y1_pred, bins=bins, density=True, color='darkred', histtype='step', label='Z=1')
    ax.hist(Y0_pred, bins=bins, density=True, color='red', histtype='step', label='Z=0')
    ax.hist(Y_1_pred, bins=bins, density=True, color='tomato', histtype='step', label='Z=-1')

    # cosmetics
    ax.legend(loc='best')
    ax.set_title('Classifier output')
    ax.set_xlabel('Classifier output f(X|Z=z)')
    
def show_clf(clf, generate, pandas=False):
    
    # create the figure
    fig, ax  = plt.subplots(1, 3, figsize=(15, 5))

    # roc ruve
    show_roc_curves(ax[0], clf, generate, pandas=pandas)
    
    # decision boundary
    show_decision_boundary(ax[1], clf, generate, pandas=pandas)
    
    # classifier output
    show_clf_output(ax[2], clf, generate, pandas=pandas)
