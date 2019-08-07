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

def test_poly_fit(N, generate):
	# training and test sets
	X, Y, Z = generate(N)
	X_test, Y_test, Z_test = generate(N)
	
	# fit the models
	clf = FairboostClassifier(Z, n_estimators=100)
	clf.fit(X, Y)
	    
	# predict
	F_test = clf.predict_proba(X_test)[:,1]
	
	# only take the signal values
	x = X_test[Y_test==1]
	f = F_test[Y_test==1]
	z = Z_test[Y_test==1]
	
	# create bins
	nbins=100
	hist, edges = np.histogram(f, range=(0,1), bins=nbins)
	centres = 0.5 * (edges[:-1] + edges[1:])
	
	# compute metrics over bins
	idx = np.digitize(f, edges)
	bin_means = [np.mean(z[idx==i]) for i in range(1, len(edges))]
	bin_stds = [np.std(z[idx==i])/np.sqrt(len(z[idx==i])) for i in range(1, len(edges))]
	
	# fit the model
	adv = PolynomialModel()
	adv.fit(f, z)
	adv.predict(f)
	adv.negative_gradient(f)
	
	# create model response
	xs = np.linspace(0,1,100)
	ys = adv.predict(xs)
	
	fig, ax = plt.subplots(figsize=(10,10))
	ax.scatter(f, z, alpha=0.1, color='darkblue')
	ax.errorbar(centres, bin_means, xerr=0.5/nbins, yerr=bin_stds, color='orange')
	ax.plot(xs, ys, color='red')
	ax.set_xlabel('classifier output')
	ax.set_ylabel('Z')
	ax.set_xlim(0,1)
	plt.show()
