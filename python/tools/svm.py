import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import inspect
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('seaborn')

"""
Original: aHR0cHM6Ly9naXN0LmdpdGh1Yi5jb20vbWJsb25kZWwvNTg2NzUz
Raw: aHR0cHM6Ly9naXN0LmdpdGh1YnVzZXJjb250ZW50LmNvbS9tYmxvbmRlbC81ODY3NTMvcmF3LzZlMGMyYWMzMTYwYWI1YTcwNjhiNmYwZDM5ZjJlMDZiOTdlYjBmMmIvc3ZtLnB5
Art: aHR0cHM6Ly9weXRob25wcm9ncmFtbWluZy5uZXQvc29mdC1tYXJnaW4ta2VybmVsLWN2eG9wdC1zdm0tbWFjaGluZS1sZWFybmluZy10dXRvcmlhbC8=

https://github.com/soloice/SVM-python
https://github.com/AFAgarap/support-vector-machine
http://sdsawtelle.github.io/blog/output/week7-andrew-ng-machine-learning-with-python.html

https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html

http://web.mit.edu/6.034/wwwbob/svm-notes-long-08.pdf

http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/
https://gist.github.com/WittmannF/60680723ed8dd0cb993051a7448f7805

https://github.com/gmum/pykernels

http://cvxopt.org/applications/svm/

http://www.cs.toronto.edu/~duvenaud/cookbook/
https://en.wikipedia.org/wiki/Kernel_(statistics)

http://people.math.umass.edu/~nahmod/Good_Kernels_PeriodicFunctions.pdf
http://www.robots.ox.ac.uk/~az/lectures/ml/lect3.pdf

Kernel Smoother: https://en.wikipedia.org/wiki/Kernel_smoother

Positive Definite Kernel: https://en.wikipedia.org/wiki/Positive-definite_kernel
- 

Stochastic Kernel: https://en.wikipedia.org/wiki/Markov_kernel
Graph Kernel: https://en.wikipedia.org/wiki/Graph_kernel
Tree Kernel: https://en.wikipedia.org/wiki/Tree_kernel
Kernel Density Estimation: https://en.wikipedia.org/wiki/Kernel_density_estimation
Heat Kernel: https://en.wikipedia.org/wiki/Heat_kernel
Poisson Kernel: https://en.wikipedia.org/wiki/Poisson_kernel
Abel's Theorem: https://en.wikipedia.org/wiki/Abel%27s_theorem
"""

class Kernels(object):

    @staticmethod
    def Linear(x, y, c=0.0):
        """Linear Kernel

        The Linear kernel is the simplest kernel function. It is given by the inner
        product <x,y> plus an optional constant c.Kernel algorithms using a linear
        kernel are often equivalent to their non-kernel counterparts,
        i.e. KPCA with linear kernel is the same as standard PCA.

        k(x, y) = x^T y + c

        Reference:
        """
        return np.dot(x, y) + c

    @staticmethod
    def Polynomial(x, y, d=3, c=1.0):
        """Inhomogeneous Polynomial Kernel

        The Polynomial kernel is a non-stationary kernel. Polynomial kernels are
        well suited for problems where all the training data is normalized.

        k(x, y) = (alpha x^T y + c)^d 
        
        Adjustable parameters are the slope alpha, the constant term c and the
        polynomial degree d. When c = 0, the kernel is called homogeneuos.

        Reference: https://en.wikipedia.org/wiki/Polynomial_kernel
        """
        return (np.dot(x, y) + c) ** d

    @staticmethod
    def PolynomialHomogeneuos(x, y, d=3):
        """Homogeneous Polynomial Kernel"""
        return Kernels.Polynomial(x, y, d, 0.0)

    @staticmethod
    def Gaussian(x, y, sigma=5.0):
        """Gaussian Kernel 

        The Gaussian kernel is an example of radial basis function kernel.

        k(x, y) = exp(-frac{ ||x-y||^2}{2 sigma^2}) 

        Alternatively, it could also be implemented using

        k(x, y) = exp(-gamma ||x-y||^2 ) 

        The adjustable parameter sigma plays a major role in the performance
        of the kernel, and should be carefully tuned to the problem at hand.
        If overestimated, the exponential will behave almost linearly and the
        higher-dimensional projection will start to lose its non-linear power.
        In the other hand, if underestimated, the function will lack regularization
        and the decision boundary will be highly sensitive to noise in training data.

        URL: https://en.wikipedia.org/wiki/Gaussian_kernel
        """
        return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

    @staticmethod
    def RBF(x, y, gamma=5.0):
        """Radial Basis Function Kernel (RBF)
        
        See: Gaussian Kernel
        
        URL: https://en.wikipedia.org/wiki/Radial_basis_function_kernel 
        """
        return np.exp(-gamma * linalg.norm(np.subtract(x, y)))

    @staticmethod
    def Exponential(x, y, sigma=1.0):
        """Exponential Kernel

        The exponential kernel is closely related to the Gaussian kernel,
        with only the square of the norm left out. It is also a radial basis
        function kernel.

        k(x, y) = exp(-frac{ ||x-y|| }{2 sigma^2}) 

        URL: 
        """
        return np.exp(-linalg.norm(x-y) / (2 * (sigma ** 2)))

    @staticmethod
    def Laplacian(x, y, sigma=1.0):
        """Laplacian Kernel

        The Laplace Kernel is completely equivalent to the exponential kernel,
        except for being less sensitive for changes in the sigma parameter.
        Being equivalent, it is also a radial basis function kernel.

        k(x, y) = exp(-frac{ ||x-y|| }{sigma})

        It is important to note that the observations made about the sigma
        parameter for the Gaussian kernel also apply to the Exponential
        and Laplacian kernels.

        URL: https://www.ml.cmu.edu/research/dap-papers/kondor-diffusion-kernels.pdf
        """
        return np.exp(-linalg.norm(x-y) / sigma)

    @staticmethod
    def Sigmoid(x, y, alpha=0.001, c=1.0):
        """Hyperbolic Tangent (Sigmoid) Kernel

        The Hyperbolic Tangent Kernel is also known as the Sigmoid Kernel and
        as the Multilayer Perceptron (MLP) kernel. The Sigmoid Kernel comes
        from the Neural Networks field, where the bipolar sigmoid function
        is often used as an activation function for artificial neurons.

        k(x, y) = tanh(alpha x^T y + c) 

        It is interesting to note that a SVM model using a sigmoid kernel
        function is equivalent to a two-layer, perceptron neural network.
        This kernel was quite popular for support vector machines due to
        its origin from neural network theory. Also, despite being only
        conditionally positive definite, it has been found to perform well
        in practice.

        There are two adjustable parameters in the sigmoid kernel, the slope
        alpha and the intercept constant c. A common value for alpha is 1/N,
        where N is the data dimension. A more detailed study on sigmoid kernels
        can be found in the works by Hsuan-Tien and Chih-Jen.

        URL: 
        """
        return np.tanh(alpha * np.dot(x,y) + c)

    @staticmethod
    def RationalQuadratic(x, y, c=1.0):
        """Rational Quadratic Kernel

        The Rational Quadratic kernel is less computationally intensive than
        the Gaussian kernel and can be used as an alternative when using the
        Gaussian becomes too expensive.

        k(x, y) = 1 - frac{ ||x-y||^2 }{ ||x-y||^2 + c } 

        URL: 
        """
        return 1 - (linalg.norm(x-y)**2 / (linalg.norm(x-y)**2 + c))

    @staticmethod
    def MultiQuadric(x, y, c=1.0):
        """Multi Quadric Kernel

        The Multiquadric kernel can be used in the same situations as the
        Rational Quadratic kernel. As is the case with the Sigmoid kernel,
        it is also an example of an non-positive definite kernel.

        k(x, y) = sqrt{ ||x-y||^2 + c^2 }

        URL: 
        """
        return np.sqrt(linalg.norm(x-y)**2 + c**2)

    @staticmethod
    def InverseMultiQuadric(x, y, theta=1.0):
        """Inverse Multi Quadric Kernel

        The Inverse Multi Quadric kernel. As with the Gaussian kernel, it
        results in a kernel matrix with full rank (Micchelli, 1986) and thus
        forms a infinite dimension feature space.

        k(x, y) = frac{ 1 }{ sqrt{ ||x-y||^2 + theta^2 } }

        URL: 
        """
        return 1 / Kernels.MultiQuadric(x, y, theta)


class SVM(object):

    def __init__(self, kernel=Kernels.Linear, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        """Fit model
        Support vector machine training using matrix completion technique.

        The basic idea is to replace the dense kernel matrix with the
        maximum determinant positive definite completion of a subset of
        the entries of the kernel matrix. The resulting approximate kernel
        matrix has a sparse inverse and this property can be exploited to
        dramatically improve the efficiency of interior-point methods

        URL: http://www.seas.ucla.edu/~vandenbe/publications/svmcmpl.pdf
        """
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            # Soft
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == Kernels.Linear:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))

if __name__ == "__main__":
    import pylab as pl

    def gen_lin_separable_data():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def gen_discrete_separable_data():
        # generate training data in the 2-d case
        mean1 = [0, -2]
        mean2 = [0, 2]
        mean3 = [3, 2]
        mean4 = [0, 2]
        cov = [[1.0,0.8], [0.8, 1.0]]
        X1 = np.random.multivariate_normal(mean1, cov, 50)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 50)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def gen_non_lin_separable_data():
        mean1 = [-1, 2]
        mean2 = [1, -1]
        mean3 = [4, -4]
        mean4 = [-4, 4]
        cov = [[1.0,0.8], [0.8, 1.0]]
        X1 = np.random.multivariate_normal(mean1, cov, 50)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 50)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def gen_lin_separable_overlap_data():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def split_train(X1, y1, X2, y2):
        X1_train = X1[:90]
        y1_train = y1[:90]
        X2_train = X2[:90]
        y2_train = y2[:90]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        return X_train, y_train

    def split_test(X1, y1, X2, y2):
        X1_test = X1[90:]
        y1_test = y1[90:]
        X2_test = X2[90:]
        y2_test = y2[90:]
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        return X_test, y_test

    def plot_margin(X1_train, X2_train, clf):
        def f(x, w, b, c=0):
            # given x, return y such that [x,y] in on the line
            # w.x + b = c
            return (-w[0] * x - b + c) / w[1]

        pl.plot(X1_train[:,0], X1_train[:,1], "ro")
        pl.plot(X2_train[:,0], X2_train[:,1], "bo")
        pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")

        # w.x + b = 0
        a0 = -4; a1 = f(a0, clf.w, clf.b)
        b0 = 4; b1 = f(b0, clf.w, clf.b)
        pl.plot([a0,b0], [a1,b1], "k")

        # w.x + b = 1
        a0 = -4; a1 = f(a0, clf.w, clf.b, 1)
        b0 = 4; b1 = f(b0, clf.w, clf.b, 1)
        pl.plot([a0,b0], [a1,b1], "k--")

        # w.x + b = -1
        a0 = -4; a1 = f(a0, clf.w, clf.b, -1)
        b0 = 4; b1 = f(b0, clf.w, clf.b, -1)
        pl.plot([a0,b0], [a1,b1], "k--")

        pl.axis("tight")
        pl.show()

    def plot_contour(X1_train, X2_train, clf):
        pl.plot(X1_train[:,0], X1_train[:,1], 'o', color='#4C72B0')
        pl.plot(X2_train[:,0], X2_train[:,1], 'o', color='#55A868')
        pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="#C44E52")

        X1, X2 = np.meshgrid(np.linspace(-6,6,50), np.linspace(-6,6,50))
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        Z = clf.project(X).reshape(X1.shape)
        pl.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

        pl.axis("tight")
        pl.show()

    def test_specific(kernel=Kernels.Linear, data=gen_lin_separable_data, C=None):
        X1, y1, X2, y2 = data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)

        clf = SVM(kernel, C=C)
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)

        cr = classification_report(y_test, y_predict)
        cm = confusion_matrix(y_test, y_predict)
        
        print("\n\nClassification report for classifier %s:\n%s\n" % (clf, cr))

        print("Confusion matrix:\n%s" % cm)

        print("%d out of %d predictions correct" % (correct, len(y_predict)))

        classes = ["1","0"]
        df_cm = pd.DataFrame(cm, index=classes, columns=classes)
        plt.figure()
        sns.heatmap(df_cm, annot=True, cmap="RdYlBu_r") #cmap="PuBuGn"
        plt.show(block=False)

        plt.figure()
        plot_contour(X_train[y_train==1], X_train[y_train==-1], clf)
 

    #test_specific(kernel=Kernels.Linear, data=gen_lin_separable_data)
    #test_specific(kernel=Kernels.Linear, data=gen_lin_separable_data, C=1.0)
    #test_specific(kernel=Kernels.Polynomial, data=gen_non_lin_separable_data)
    test_specific(kernel=Kernels.Gaussian, data=gen_non_lin_separable_data)
    #test_specific(kernel=Kernels.Sigmoid, data=gen_discrete_separable_data)
    #test_specific(kernel=Kernels.RationalQuadratic, data=gen_non_lin_separable_data)
    #test_specific(kernel=Kernels.MultiQuadric, data=gen_non_lin_separable_data)
    #test_specific(kernel=Kernels.InverseMultiQuadric, data=gen_non_lin_separable_data)
