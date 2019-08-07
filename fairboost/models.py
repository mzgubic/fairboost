import scipy.optimize as spo

class PolynomialModel:
    """
    Works only for one dimensional X...
    """

    def __init__(self, order=3):
        
        self.order = order
        self.coefficients = np.ones(shape=self.order+1)
    
    def fit(self, X, Y):

        # prepare objective
        def loss(coefficients):
            output = self._forward(X, *coefficients)
            return np.mean((Y-output)**2)
        
        # fit parameters
        res = spo.minimize(loss, x0=np.ones(len(self.coefficients)))
        
        # save as coefficients
        self.coefficients = res.x
    
    def _forward(self, X, *coefficients):
        
        # loop over coefficients
        result = np.zeros(shape=X.shape)
        for i, a in enumerate(coefficients):
            result += - a * X**i
        return result
    
    def predict(self, X):
        """
        A(x) = a_0 + a_1*x + a_2*x**2 + ... + a_n*x**n
        """
        res = self._forward(X, *self.coefficients)
        return res
    
    def negative_gradient(self, X):
        """
        - A'(x) = - (a_1 + 2*a_2*x + 3*a_3*x**2 + n*a_n*x**(n-1))
        """
        # loop over coefficients
        result = np.zeros(shape=X.shape)
        for i, a in enumerate(self.coefficients):
            result += i * a * X**(i-1)
        return result


