import time
import math
import cvxopt
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from re import A
from enum import unique
from scipy import optimize
from cProfile import label
from itertools import cycle
from matplotlib import gridspec
from collections import OrderedDict
from sklearnex import patch_sklearn
from sklearn.metrics import accuracy_score, cohen_kappa_score
patch_sklearn()
    

# Stochastic distances
def distances(df, label_column, c1, c2, method='JM'):
    # Richards (2013, p. 350)
    # https://github.com/KolesovDmitry/i.jmdist/blob/master/i.jmdist
    # https://jneuroengrehab.biomedcentral.com/articles/10.1186/s12984-017-0283-5/tables/2

    def mahalanobis_distance(m1, m2, cov1, cov2):
        m = (m1 - m2)
        cov = (cov1 + cov2)/2
        inv_cov = np.linalg.inv(cov)

        tmp = np.core.dot(m.T, inv_cov)
        tmp = np.core.dot(tmp, m)
        
        MH = np.sqrt(tmp)
        return MH


    def bhattacharyya_distance(m1, m2, cov1, cov2):
        
        m = (m1 - m2)
        cov = (cov1 + cov2)/2
        inv_cov = np.linalg.inv(cov)

        pt1 = np.core.dot(m.T, inv_cov)
        pt1 = np.core.dot(pt1, m)
        pt1 = pt1/8.0
        
        pt2 = np.linalg.det(cov) / np.sqrt( np.linalg.det(cov1)*np.linalg.det(cov2) )
        pt2 = np.log(pt2)
        pt2 = pt2/2.0

        B = pt1 + pt2
        return B


    def jeffries_matusita_distance(m1, m2, cov1, cov2):
        # JM = 2(1-np.e(-B)) :: for normally distributed classes
        B = bhattacharyya_distance(m1, m2, cov1, cov2)

        #JM = np.sqrt(2 * (1 - np.exp(-B)))
        JM = 2 * (1 - np.exp(-B))
        return JM

    df_np = df.to_numpy()
    col_index_dict = dict(zip(df.columns, list(range(0,len(df.columns)))))

    classes = np.unique(df_np[:, col_index_dict[label_column]])

    if c1 not in classes or c2 not in classes:
        raise ValueError(
            "Both c1 and c2 must be amongst the object classes."
        )

    #x_c1 = df.loc[df[label_column]==c1, df.columns!=label_column]
    #x_c2 = df.loc[df[label_column]==c2, df.columns!=label_column]

    cols = [v for k, v in col_index_dict.items() if k != label_column]
    x_c1 = df_np[(df_np[:, col_index_dict[label_column]] == c1)][:, cols]
    x_c2 = df_np[(df_np[:, col_index_dict[label_column]] == c2)][:, cols]

    #mean_c1 = np.array(np.mean(x_c1))
    #mean_c2 = np.array(np.mean(x_c2))

    mean_c1 = np.array(np.mean(x_c1, axis=0)) # axis=0 -> average by column
    mean_c2 = np.array(np.mean(x_c2, axis=0)) # axis=0 -> average by column

    cov_c1 = np.cov(x_c1, rowvar=False)
    cov_c2 = np.cov(x_c2, rowvar=False)


    if method == 'Bhattacharyya' or method == 'B':
        dist = bhattacharyya_distance(mean_c1, mean_c2, cov_c1, cov_c2)

    elif method == 'Jeffries-Matusita' or method == 'JM':
        dist = jeffries_matusita_distance(mean_c1, mean_c2, cov_c1, cov_c2)

    elif method == 'Mahalanobis' or method == 'M':
        dist = mahalanobis_distance(mean_c1, mean_c2, cov_c1, cov_c2)
        
    else:
        raise ValueError(
                "Method not found.\n"
                + "The implemented stochastic distances are: Bhattacharyya; Jeffries-Matusita (JM); and, Mahalanobis."
            )
    
    return dist


def class_separability_analysis(df, label_column, method='JM'):

    d = OrderedDict()
    for i, (c1, c2) in enumerate(list(itertools.combinations(df[label_column].unique(), 2) )):
        dist = distances(df, label_column, c1=c1, c2=c2, method=method)

        d[i] = {"pair":i, "c1":c1, "c2":c2, "distance":dist}

    df_class_separability = pd.DataFrame.from_dict(d, "index")
    df_class_separability.sort_values(by=['distance'], ascending=True, inplace=True)
    
    return df_class_separability


"""
# NOTE: REFERENCE
- based on the mlxtend library
https://github.com/rasbt/mlxtend/blob/8c61c063f98f0d9646edfd8b0270b77916f0c434/mlxtend/utils/checking.py
"""
def format_kwarg_dictionaries(default_kwargs=None, user_kwargs=None,
                              protected_keys=None):
    """Function to combine default and user specified kwargs dictionaries
    Parameters
    ----------
    default_kwargs : dict, optional
        Default kwargs (default is None).
    user_kwargs : dict, optional
        User specified kwargs (default is None).
    protected_keys : array_like, optional
        Sequence of keys to be removed from the returned dictionary
        (default is None).
    Returns
    -------
    formatted_kwargs : dict
        Formatted kwargs dictionary.
    """
    formatted_kwargs = {}
    for d in [default_kwargs, user_kwargs]:
        if not isinstance(d, (dict, type(None))):
            raise TypeError('d must be of type dict or None, but '
                            'got {} instead'.format(type(d)))
        if d is not None:
            formatted_kwargs.update(d)
    if protected_keys is not None:
        for key in protected_keys:
            formatted_kwargs.pop(key, None)

    return formatted_kwargs


class SVM:
    """
    # NOTE: CODE REFERENCES
    
    Book ML refined (Watt, 2020): 
        https://github.com/jermwatt/machine_learning_refined
        https://github.com/jermwatt/machine_learning_refined/blob/gh-pages/notes/7_Linear_multiclass_classification/7_2_OvA.ipynb
        https://github.com/jermwatt/machine_learning_refined/blob/gh-pages/mlrefined_libraries/superlearn_library/ova_illustrator.py
            - Toy dataset
            - Colors
            - Functions plot_margin (adapted), show_dataset (adapted)
            - Also used code from the functions in gh: plot_data, plot_all_separators, region_coloring, 
                point_and_projection

    Python library: mlxtend (machine learning extensions)
        https://github.com/rasbt/mlxtend/blob/master/mlxtend/plotting/decision_regions.py
        https://github.com/rasbt/mlxtend/blob/8c61c063f98f0d9646edfd8b0270b77916f0c434/mlxtend/utils/checking.py
            - Multiclass boundary
            - Função format_kwarg_dictionaries

    Dr. Ian Gregory - GitHub
        https://github.com/DrIanGregory/MachineLearning-SupportVectorMachines/blob/master/supportVectorMachines.py
        https://github.com/DrIanGregory/MachineLearning-SupportVectorMachines/blob/master/plotting.py
            - SVM class structure
            - Kernel functions (linear, gaussian, polynomial)
            - Functions binary_fit, project_features, project (adapted), plot_margin (adapted)

    Kolesov Dmitry - GitHub
        https://github.com/KolesovDmitry/i.jmdist/blob/master/i.jmdist
            - Mahalanobis distance
            - Bhattacharyya distance
            - Jeffries-Matusita distance
    """

    def __init__(self, kernel='linear', C=0, gamma='scale', degree=3, coef0=0.5, strategy='OAA', 
                    AGL_default_strategy='OAA', colors=None, verbose=True, 
                    verbose_opt=False, minx=None, maxx=None, miny=None, maxy=None,
                    tol=None, abstol=None, reltol=None, feastol=None, qp_solver = 'cvxopt'):
        
        #warnings.filterwarnings("ignore")
        warnings.filterwarnings("ignore", message="No contour levels were found within the data range.")
        #warnings.filterwarnings( "ignore", module = "matplotlib\..*" )

        self.time = []

        # class stats
        self.classes = []
        self.class_mean = []
        self.class_cov = []

        self.n_classes = None
        self.AGL_n_classes = None

        # list of data and labels
        self.dim = None
        self.x = []
        self.y = []

        self.c1 = []
        self.c2 = []

        # list of merged classes [(a,b), (c,e)]
        self.superclasses = dict()
        self.superclass_tests = []
        self.superclass_tests_summary = []
        self.superclass_classifiers = []
        self.superclass_per_classifier = []

        # colors
        if colors is None:
            #self.colors = [
            #    [0, 0.4, 1], [1, 0, 0.4], [0, 1, 0.5], [1, 0.7, 0.5], [0.7, 1, 0.5], [0.5, 0.7, 1], [0.7, 0.6, 0.5], 
            #    'mediumaquamarine', [0.2, 0.2, 0.2], [0.7, 0.7, 0.7], [0.9, 0.3, 0.2]
            #]
            self.colors = ['blue', 'red', 'darkgreen', 'goldenrod', 'cornflowerblue', 'lightcoral', 'springgreen', 
                            'cyan', 'lightsalmon', 'lime', 'darkcyan', 'lightslategray', 'blueviolet', 'magenta',
                            'pink', 'silver', 'black']
        else:
            self.colors = colors

        # Kernel
        if kernel is None:
            kernel = 'linear'
        self.kernel = kernel

        # Quadratic-problem solver
        if qp_solver is None:
            qp_solver == 'cvxopt'
        self.qp_solver = qp_solver

        # other parameters
        self.params = None

        if C is None:
            C = 0
        self.C = float(C)
        self.strategy = strategy
        self.AGL_default_strategy = AGL_default_strategy
        self.verbose = verbose
        self.verbose_opt = verbose_opt
        self.gamma = gamma # Only significant in kernels 'gausssian', 'polynomial' and 'sigmoid'
        self.degree = degree # Only significant in kernel 'polynomial'
        self.coef0 = coef0 # Only significant in kernels 'polynomial' and 'sigmoid'
        
        self.K = []
        self.alphas = []
        self.sv_bool = []
        self.sv = []
        self.sv_y = []

        self.free_sv_alphas = []
        self.free_sv_bool = []
        self.free_sv = []
        self.free_sv_y = []

        self.b = []
        self.w = []
        self.intercept = []

        self.margins = []

        # Tolerances
        self.tol = tol
        # Absolute accuracy (default: 1e-7)
        self.abstol = abstol
        # Relative accuracy (default: 1e-6)
        self.reltol = reltol
        # Tolerance for feasibility conditions (default: 1e-7)
        self.feastol = feastol
        
        self.solution = []
        
        self.f1 = None
        self.f2 = None
        self.xx = None
        self.yy = None
        self.Z = None

        if (minx is None or maxx is None or miny is None or maxy is None):
            self.minx = None
            self.maxx = None
            self.miny = None
            self.maxy = None
        else:
            self.minx = minx
            self.maxx = maxx
            self.miny = miny
            self.maxy = maxy

    #https://www.researchgate.net/publication/299999664_Support_Vector_Machine_-_A_Large_Margin_Classifier_to_Diagnose_Skin_Illnesses
    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2)

    def polynomial_kernel(self, x, y, gamma=1, coef0=0.5, degree=3):
        # Inputs:
        #   x       : vector of x data.
        #   y       : vector of y data.
        #   gamma   : 
        #   coef0   : is a constant                     <==
        #   degree  : is the order of the polynomial.
        return (gamma*np.dot(x, y) + coef0) ** degree

    def rbf_kernel(self, x, y, gamma=0.5):
        return np.exp(-gamma*np.linalg.norm(x - y) ** 2 )
    
    #https://www.researchgate.net/publication/305356950_Evaluation_of_Nitric_Oxide_in_Lacunar_Stroke_and_Young_Healthy_during_Cerebrovascular_Reactivity_by_Support_Vector_Machine/figures?lo=1
    def sigmoidal_kernel(self, x, y, gamma=1, coef0=0.5):
        return math.tanh(gamma*np.dot(x, y) + coef0)

    
    def binary_fit(self, x, y):
        # Fit the SVM model according to a given training data set
        # Inputs:
        #   x   : vector of data features of shape (n_samples, n_features).
        #   y   : vector of labels ({+1, -1}) of shape (n_samples, ).
        # Returns:
        #   self   : Fitted estimator
        
        start = time.process_time()

        def is_positive_definite(B):
            """Returns true when input is positive-definite, via Cholesky"""
            try:
                _ = np.linalg.cholesky(B)
                return True
            except np.linalg.LinAlgError:
                return False

        def nearest_positive_definite_matrix(A):
            ## REFERENCE: https://math.stackexchange.com/questions/3513035/understanding-and-implementing-the-support-vector-machine-algorithm
            """
            Find the nearest positive-definite matrix to input

            A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
            credits [2].

            [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

            [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
            matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
            """
            B = (A + A.T) / 2
            _, s, V = np.linalg.svd(B)
            H = np.dot(V.T, np.dot(np.diag(s), V))
            A2 = (B + H) / 2
            A3 = (A2 + A2.T) / 2

            if is_positive_definite(A3):
                return A3

            spacing = np.spacing(np.linalg.norm(A))
            # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
            # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
            # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
            # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
            # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
            # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
            # `spacing` will, for Gaussian random matrixes of small dimension, be on
            # othe order of 1e-16. In practice, both ways converge, as the unit test
            # below suggests.
            I = np.eye(A.shape[0])
            k = 1
            while not is_positive_definite(A3):
                mineig = np.min(np.real(np.linalg.eigvals(A3)))
                A3 += I * (-mineig * k**2 + spacing)
                k += 1

            return A3

        
        def solve_quadratic_opt_SLSQP(self, x, y, n_samples):
            """
            N: n_samples
            """
            # force to float type
            y = y * 1.0

            # Lagrange dual problem
            def Ld0(G, alpha):
                return alpha.sum() - 0.5 * alpha.dot(alpha.dot(G))

            # Partial derivate of Ld on alpha
            def Ld0dAlpha(G, alpha):
                return np.ones_like(alpha) - alpha.dot(G)
            
            if (self.kernel is not None and self.kernel != 'linear') and (self.C is None):
                # 'C = None' not accepted for Kernel SVM
                self.C = 0

            if (self.kernel is None or self.kernel == 'linear'):
                
                # Gram matrix of (X.y)
                Xy = x * y[:, np.newaxis]
                GramXy = np.matmul(Xy, Xy.T)

                if self.C is None or self.C == 0:
                    # Hard-margin SVM

                    # Constraints on alpha of the shape :
                    # -  d - C*alpha  = 0
                    # -  b - A*alpha >= 0
                    A = -np.eye(n_samples)
                    b = np.zeros(n_samples)
                    constraints = ({'type': 'eq',   'fun': lambda a: np.dot(a, y), 'jac': lambda a: y},
                                {'type': 'ineq', 'fun': lambda a: b - np.dot(A, a), 'jac': lambda a: -A})

                    # Maximize by minimizing the opposite
                    optRes = optimize.minimize(fun=lambda a: -Ld0(GramXy, a),
                                            x0=np.ones(n_samples), 
                                            method='SLSQP', 
                                            jac=lambda a: -Ld0dAlpha(GramXy, a), 
                                            constraints=constraints)
                    alphas = optRes.x
                
                else:
                    # Soft-Margin SVM

                    # Constraints on alpha of the shape :
                    # -  d - C*alpha  = 0
                    # -  b - A*alpha >= 0
                    A = np.vstack((-np.eye(n_samples), np.eye(n_samples)))             # <---
                    b = np.hstack((np.zeros(n_samples), self.C * np.ones(n_samples)))  # <---
                    constraints = ({'type': 'eq',   'fun': lambda a: np.dot(a, y), 'jac': lambda a: y},
                                {'type': 'ineq', 'fun': lambda a: b - np.dot(A, a), 'jac': lambda a: -A})

                    # Maximize by minimizing the opposite
                    optRes = optimize.minimize(fun=lambda a: -Ld0(GramXy, a),
                                            x0=np.ones(n_samples), 
                                            method='SLSQP', 
                                            jac=lambda a: -Ld0dAlpha(GramXy, a), 
                                            constraints=constraints)
                    alphas = optRes.x
                    
                return alphas, GramXy
            else:
                # Kernel SVM
                
                if self.kernel=='rbf':
                    hXX = np.apply_along_axis(lambda x1 : np.apply_along_axis(lambda x2:  self.rbf_kernel(x1, x2, self.gamma), 1, x), 1, x)

                elif self.kernel == 'polynomial':
                    hXX = np.apply_along_axis(lambda x1 : np.apply_along_axis(lambda x2:  self.polynomial_kernel(x1, x2, self.gamma, self.coef0, self.degree), 1, x), 1, x)
                
                elif self.kernel == 'sigmoid':
                    hXX = np.apply_along_axis(lambda x1 : np.apply_along_axis(lambda x2:  self.sigmoidal_kernel(x1, x2, self.gamma, self.coef0), 1, x), 1, x)
                
                else:
                    raise ValueError(f"Kernel function not found for '{self.kernel}'")

                yp = y.reshape(-1, 1)
                GramHXy = hXX * np.matmul(yp, yp.T) 

                # Constraints on alpha of the shape :
                # -  d - C*alpha  = 0
                # -  b - A*alpha >= 0
                A = np.vstack((-np.eye(n_samples), np.eye(n_samples)))             # <---
                b = np.hstack((np.zeros(n_samples), self.C * np.ones(n_samples)))  # <---
                constraints = ({'type': 'eq',   'fun': lambda a: np.dot(a, y),     'jac': lambda a: y},
                                {'type': 'ineq', 'fun': lambda a: b - np.dot(A, a), 'jac': lambda a: -A})

                # Maximize by minimizing the opposite
                optRes = optimize.minimize(fun=lambda a: -Ld0(GramHXy, a),
                                        x0=np.ones(n_samples), 
                                        method='SLSQP', 
                                        jac=lambda a: -Ld0dAlpha(GramHXy, a), 
                                        constraints=constraints)
                alphas = optRes.x
            
                return alphas, GramHXy


        def solve_quadratic_opt(self, x, y, n_samples):
            
            # force to float type
            y = y * 1.0

            # Gram matrix: G_ij = <v_i, v_j>
            # the matrix of the inner product of each vector and its corresponding vectors in same
            K = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(n_samples):

                    # Kernel trick.
                    if self.kernel == 'linear':
                        K[i, j] = self.linear_kernel(x[i], x[j])

                    elif self.kernel=='rbf':
                        K[i, j] = self.rbf_kernel(x[i], x[j], self.gamma)   # Kernel trick.
                        #self.C = None   # Not used in rbf kernel.

                    elif self.kernel=='sigmoid':
                        K[i, j] = self.sigmoidal_kernel(x[i], x[j], self.gamma, self.coef0)   # Kernel trick.

                    elif self.kernel == 'polynomial':
                        K[i, j] = self.polynomial_kernel(x[i], x[j], self.gamma, self.coef0, self.degree)

            # Converting into cvxopt format for optimization
            P = cvxopt.matrix(np.outer(y, y) * K)
            q = cvxopt.matrix(np.ones(n_samples) * -1.)
            #A = cvxopt.matrix(y, (1, n_samples))
            #A = cvxopt.matrix(A, (1, n_samples), 'd')
            A = cvxopt.matrix(y, (1, n_samples))
            b = cvxopt.matrix(0.0)

            
            if self.C is None or self.C==0:
                G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1.))
                h = cvxopt.matrix(np.zeros(n_samples))
            
            # soft-margin svm
            else: 
                # Restricting the optimisation with parameter C.
                # Constraint 1
                tmp1 = np.diag(np.ones(n_samples) * -1.)
                # Constraint 2
                tmp2 = np.identity(n_samples)
                G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
                # Constraint 1
                tmp1 = np.zeros(n_samples)
                # Constraint 2
                tmp2 = np.ones(n_samples) * self.C
                h = cvxopt.matrix(np.hstack((tmp1, tmp2)))
            
            # SV tolerance (epsilon)
            if self.tol is None:
                tol = 1e-5
                self.tol = tol
            else:
                tol = self.tol
            
            if self.abstol is None:
                abstol = 1e-10
                self.abstol = abstol
            else:
                abstol = self.abstol

            if self.reltol is None:
                reltol = 1e-10
                self.reltol = reltol
            else:
                reltol = self.reltol
                
            if self.feastol is None:
                feastol = 1e-10
                self.feastol = feastol
            else:
                feastol = self.feastol

            #tol = 1e-5
            #try:
            #    P_array = np.array(P)
            #    if not is_positive_definite(P_array):
            #        P = nearest_positive_definite_matrix(P_array)
            #        P = cvxopt.matrix(P)
            #        tol=0
            #
            #    G_array = np.array(G)
            #    if not is_positive_definite(G_array):
            #        G = nearest_positive_definite_matrix(G_array)
            #        G = cvxopt.matrix(G)
            #        tol=0
            #except Exception as err:
            #    print("Error to determine the nearest positive definite matrix: ", err)
            
            # Setting options:
            # The tolerance parameters define how much variation we will be allowing before declaring convergence. 
            cvxopt.solvers.options['show_progress'] = self.verbose_opt
            cvxopt.solvers.options['abstol'] = abstol
            cvxopt.solvers.options['reltol'] = reltol
            cvxopt.solvers.options['feastol'] = feastol

            # Solve the Quadratic Programming (QP) problem:
            self.P = P
            self.q = q
            self.G = G
            self.h = h
            self.A = A
            self.bb = b
            solution = cvxopt.solvers.qp(P, q, G, h, A, b)
            
            self.solution.append(solution)

            # Lagrange multipliers
            # Flatten the matrix into a vector of all the Langrangian multipliers.
            alphas = np.ravel(solution['x'])

            n_svs = len(alphas[alphas > tol]) # > 0.00001
            #print(f"Tol: {tol} ; Número inicial de SVs: {n_svs}")

            #tol_step = 1e-10
            #while n_svs < 1 and tol > tol_step:
            #    tol -= tol_step
            #    n_svs = len(alphas[alphas > tol])
            #    
            #    if not (n_svs < 1 and tol > tol_step):
            #        print(f"Tol: {tol} ; Número de SVs: {n_svs}")

            return alphas, K#, tol


        if not all(hasattr(self, attr) for attr in ['x', 'y', 'kernel', 'C', 'verbose', 'verbose_opt', 'strategy']):
            raise ValueError(
                "The object does not contain all required attributes. \n"
                + "Initialize the object before fitting."
            )
            
        if not isinstance(self.x, list) or not isinstance(self.y, list):
            raise ValueError(
                "The object attributes self.x and self.y should be lists. \n"
                + "Initialize the object before fitting." 
            )
            
        self.x.append(x)
        self.y.append(y)

        n_samples, n_features = x.shape

        if n_samples != y.shape[0]:
            raise ValueError(
                "x and y have incompatible shapes.\n"
                + "x has %s samples, but y has %s." % (n_samples, y.shape[0])
            )

        # SV tolerance (epsilon)
        if self.tol is None:
            tol = 1e-5
            self.tol = tol
        else:
            tol = self.tol

        if self.qp_solver == 'slsqp':
            # Sequential Least Squares Programming (SLSQP)
            """We found some problems when using Kernel SVM with very small C (~0). 
            The cvxopt solver performed better in those case, even thought it defines most samples as SVs when C is very small"""
            try:
                alphas, K = solve_quadratic_opt_SLSQP(self, x, y, n_samples)
            except:
                alphas, K = solve_quadratic_opt(self, x, y, n_samples)
                self.qp_solver = 'cvxopt'
        else: #if self.qp_solver == 'cvxopt':
            # Sometimes, the cvxopt returns the following error: Rank(A) < p or Rank([P; A; G]) < n
            try:
                alphas, K = solve_quadratic_opt(self, x, y, n_samples)
            except:
                alphas, K = solve_quadratic_opt_SLSQP(self, x, y, n_samples)
                self.qp_solver = 'slsqp'
        
        self.K.append(K)

        # Support vectors have values of lagrange multipliers greater than zero
        sv = alphas > tol #1e-5 #0
        ind = np.arange(len(alphas))[sv]
        self.alphas.append(alphas[sv])
        self.sv_bool.append(sv)
        sv_x = x[sv]
        self.sv.append(sv_x)
        sv_y = y[sv]
        self.sv_y.append(sv_y)

        if self.C > 0:
            free_sv = np.logical_and(alphas > tol, alphas < self.C) #1e-5 #0
            free_sv_ind = np.arange(len(alphas))[free_sv]
            self.free_sv_alphas.append(alphas[free_sv])
            self.free_sv_bool.append(free_sv)
            free_sv_x = x[free_sv]
            self.free_sv.append(free_sv_x)
            free_sv_y = y[free_sv]
            self.free_sv_y.append(free_sv_y)

        # Bias (For linear it is the intercept):
        # http://www.cs.cmu.edu/~guestrin/Class/10701-S07/Slides/kernels.pdf
        # bias = y_k - \sum z_i y_i  K(x_k, x_i)
        # Thus we can just predict an example with bias of zero, and
        # compute error.
        
        b = 0
        # hard margin
        if (self.C is None or self.C == 0):# or self.kernel != 'linear':
            n_sv = len(alphas[sv])
            for n in range(n_sv):
                # For all support vectors:
                b += sv_y[n]
                b -= np.sum(alphas[sv] * sv_y * K[ind[n], sv])
            
            if n_sv > 0:
                b = b / n_sv
            else:
                b = 0
            #print(f"n_sv: {n_sv}, b: {b}")
        
        # soft margin
        else:
            n_fsv = len(alphas[free_sv])
            for n in range(n_fsv):
                # For all support vectors:
                b += free_sv_y[n]
                b -= np.sum(alphas[free_sv] * free_sv_y * K[free_sv_ind[n], free_sv])
            
            if n_fsv > 0:
                b = b / n_fsv
            else:
                b = 0
            #print(f"n_fsv: {n_fsv}, b: {b}")

        #b = (np.sum(free_sv_y) - np.sum(np.ravel(alphas[free_sv] * free_sv_y * K[free_sv_ind[n], free_sv]))) / len(alphas[free_sv])

        self.b.append(b)

        # Weight vector
        if self.kernel == 'linear':
            w = np.zeros(n_features)
            for n in range(len(alphas[sv])):
                w += alphas[sv][n] * sv_y[n] * sv_x[n]

            # hard margin
            if self.C is None or self.C == 0:
                intercept = sv_y[0] - np.matmul(sv_x[0].T, w)
            else:
                signedDist = np.matmul(sv_x, w)
                minDistArg = np.argmin(signedDist)
                intercept = sv_y[minDistArg] - signedDist[minDistArg]
        else:
            w = None
            intercept = None

        self.w.append(w)
        self.intercept.append(intercept)

        self.time.append({'function': 'binary_fit', 'time': time.process_time() - start, 'msg': None})


    def OAO_strategy(self, df, label_column, figsize, aspect, exclude_cols=[]):
        
        unique_labels = df[label_column].unique()

        if label_column not in exclude_cols:
            exclude_cols.append(label_column)
        if 'temp_label' not in exclude_cols:
            exclude_cols.append('temp_label')

        ## the order matters. Both (a,b) and (b,a) are analysed.
        #for i, (c1, c2) in enumerate([(c1, c2) for c1, c2 in itertools.product(unique_labels, unique_labels) if c1 != c2] ):

        ## the order does not matter. If (a,b) exists, (b, a) is not created.
        for i, (c1, c2) in enumerate(list(itertools.combinations(unique_labels, 2) )):
            start = time.process_time()
            if self.verbose:
                print(f"OAO_strategy: class {c1} x class {c2}")
            self.c1.append(c1)
            self.c2.append(c2)

            df_ = df[df[label_column].isin([c1, c2])].copy()
            df_.loc[df_[label_column]==c1, 'temp_label'] = +1
            df_.loc[df_[label_column]==c2, 'temp_label'] = -1

            ## data
            # feature values: all columns but the last
            #x = df_.iloc[:,:-1].to_numpy()
            x = df_.loc[:, ~df_.columns.isin(exclude_cols)].to_numpy()
            # labels: only the last column
            #y = df_.iloc[:,-1].to_numpy()
            y = df_.loc[:, df_.columns == 'temp_label'].to_numpy().ravel()
            
            self.binary_fit(x, y)

            if self.dim == 2:
                self.plot_margin(i, x[y == 1], x[y == -1], c1, c2, figsize=figsize, aspect=aspect)
            
            self.time.append({'function': 'OAO_strategy', 'time': time.process_time() - start, 'msg': f"OAO_strategy: {c1} x {c2} + binary_fit"})


    def OAA_strategy(self, df, label_column, figsize, aspect, exclude_cols=[]):
        
        unique_labels = df[label_column].unique()
        
        if label_column not in exclude_cols:
            exclude_cols.append(label_column)
        if 'temp_label' not in exclude_cols:
            exclude_cols.append('temp_label')

        for i, c1 in enumerate(unique_labels):
            
            start = time.process_time()
            
            if self.verbose:
                print(f"OAA_strategy: class {c1} x rest")
            self.c1.append(c1)
            self.c2.append('rest')

            df_ = df.copy()
            df_.loc[df_[label_column]==c1, 'temp_label'] = +1
            df_.loc[df_[label_column]!=c1, 'temp_label'] = -1

            ## data
            # feature values: all columns but the last
            #x = df.iloc[:,:-1].to_numpy()
            x = df_.loc[:, ~df_.columns.isin(exclude_cols)].to_numpy()
            # labels: only the last column
            #y = df_.iloc[:,-1].to_numpy()
            y = df_.loc[:, 'temp_label'].to_numpy().ravel()
            
            if self.verbose:
                print(f" binary fit...")
            self.binary_fit(x, y)

            if self.dim == 2:
                if self.verbose:
                    print(f" plotting margin...")
                self.plot_margin(i, x[y == 1], x[y == -1], c1, figsize=figsize, aspect=aspect)
            
            self.time.append({'function': 'OAA_strategy', 'time': time.process_time() - start, 'msg': f"OAA_strategy: {c1} x rest + binary_fit"})


    def binary_svm(self, df, label_column, figsize, aspect, exclude_cols=[]):
        # two-class classification problem

        start = time.process_time()

        unique_labels = df[label_column].unique()

        if label_column not in exclude_cols:
            exclude_cols.append(label_column)
        if 'temp_label' not in exclude_cols:
            exclude_cols.append('temp_label')

        c1 = unique_labels[0]
        c2 = unique_labels[1]
        if self.verbose:
            print(f"Two-class classification problem: class {c1} x class {c2}")

        self.c1.append(c1)
        self.c2.append(c2)

        df.loc[df[label_column]==c1, 'temp_label'] = +1
        df.loc[df[label_column]==c2, 'temp_label'] = -1

        ## data
        # feature values: all columns but the last
        #x = df.iloc[:,:-1].to_numpy()
        x = df.loc[:, ~df.columns.isin(exclude_cols)].to_numpy()
        # labels: only the last column
        #y = df.iloc[:,-1].to_numpy()
        y = df.loc[:, 'temp_label'].to_numpy().ravel()
        
        #df.drop(exclude_cols, axis=1, inplace=True)

        if self.verbose:
            print(f" binary fit...")
        self.binary_fit(x, y)

        if self.dim == 2:
            if self.verbose:
                print(f" plotting margin...")
            self.plot_margin(0, x[y == 1], x[y == -1], c1, c2, figsize=figsize, aspect=aspect)

        self.time.append({'function': 'binary_svm', 'time': time.process_time() - start, 'msg': f"Binary SVM: {c1} x {c2} + binary_fit"})
        

    def fit(self, df, label_column=None, f1=0, f2=1, figsize=(12,5), aspect='auto', distance_method='JM', 
            AGL_dist_threshold=2, AGL_default_strategy='OAA', tunning_OA_threshold=1, validation_set=None,
            list_kernels=None, list_C=None, list_gammas=None, list_degrees=None):
        """
            aspect: auto, equal
            AGL_default_strategy: OAO, OAA
        """
        if label_column is None:
            raise ValueError(
                "Identify the column of labels.\n"
                + "The label_column parameter cannot be None."
            )
        if self.strategy == 'AGL' and validation_set is None:
            validation_set = df.copy()
            #raise ValueError(
            #    "Validation data set required for the AGL strategy.\n"
            #    + "The validation_set parameter cannot be None (dataframe required)."
            #)
        if tunning_OA_threshold is None:
            tunning_OA_threshold = 1
        self.tunning_OA_threshold = float(tunning_OA_threshold)

        X = df.loc[:, df.columns != label_column].to_numpy()
        Y = df.loc[:, df.columns == label_column].to_numpy().ravel()

        self.X_cols = df.columns.drop(label_column).values
        self.X = X
        self.Y = Y
        self.dim = X.shape[1]
        self.f1 = f1
        self.f2 = f2

        if self.kernel != 'linear':
            if not isinstance(self.gamma, (int, np.long, float, complex)):
                if self.gamma == 'auto':
                    # 'auto' gamma = 1 / n_features
                    self.gamma = 1/self.dim
                elif self.gamma == 'scale':
                    # 'scale' gamma = 1 / (n_features * X.var())
                    self.gamma = 1/(self.dim * self.X.var())
                else:
                    raise ValueError(f"Value of '{self.kernel}' not accepted for the gamma parameter")

        # determine plotting ranges
        if (self.minx is None or self.maxx is None or
            self.miny is None or self.maxy is None):
            #minx = min(min(self.X[:,0]), min(self.X[:,1]))
            minx = min(X[:,f1])
            miny = min(X[:,f2])
            #maxx = max(max(self.X[:,0]), max(self.X[:,1]))
            maxx = max(X[:,f1])
            maxy = max(X[:,f2])
            gapx = (maxx - minx) * 0.1
            gapy = (maxy - miny) * 0.1
            minx -= gapx
            miny -= gapy
            maxx += gapx
            maxy += gapy

            self.minx = minx
            self.maxx = maxx
            self.miny = miny
            self.maxy = maxy
        else:
            minx = self.minx
            maxx = self.maxx
            miny = self.miny
            maxy = self.maxy

        unique_labels = df[label_column].unique()
        n_classes = len(unique_labels)
        self.classes = unique_labels
        
        # Statistics: mean and covariance matrix
        start = time.process_time()
        self.class_mean2 = [np.array(np.mean(df.loc[df[label_column]==c, df.columns!=label_column])) for c in unique_labels]
        self.class_cov2 = [np.cov(df.loc[df[label_column]==c, df.columns!=label_column], rowvar=False) for c in unique_labels]
        #self.class_stats2 = time.process_time() - start
        self.time.append({'function': 'class stats 2', 'time': time.process_time() - start, 'msg': f"class_stats2"})

        start = time.process_time()
        for c in unique_labels:
            xx = df.loc[df[label_column]==c, df.columns!=label_column]
        
            #self.classes.append(c)
            self.class_mean.append(np.array(np.mean(xx)))
            #np.cov(xx, rowvar=False) :: Each col is a variable and each row is an observation.
            self.class_cov.append(np.cov(xx, rowvar=False)) 
        #self.class_stats = time.process_time() - start
        self.time.append({'function': 'class stats', 'time': time.process_time() - start, 'msg': f"class_stats"})

        if n_classes > 2:
            # One-Against-One (OAO) strategy
            if self.strategy == 'OAO':
                #self.OAO_strategy(df.copy(), label_column, figsize, aspect)
                self.OAO_strategy(df, label_column, figsize, aspect)
            
            # One-Against-All (OAA) strategy
            elif self.strategy == 'OAA':
                #self.OAA_strategy(df.copy(), label_column, figsize, aspect)
                self.OAA_strategy(df, label_column, figsize, aspect)

            # Aglomerative (AGL) strategy
            elif self.strategy == 'AGL':
                """
                REVIEW AGL strategy
                - merge pairs of classes close in the available feature space (distance <= AGL_dist_threshold)
                - perform initial classification (OAA), considering classes eventually in a merged form, as extended classes
                - perform an OAO classificaiton of each merged class, testing multiple classifiers and parameters
                - combine the results
                """
                self.AGL_default_strategy = AGL_default_strategy

                ## --- CLASS SEPARABILITY ANALYSIS ---
                df_class_separability = self.class_separability_analysis(method=distance_method)

                # pairs of classes where distance <= AGL_dist_threshold
                df_class_separability = df_class_separability.loc[df_class_separability['distance'] <= AGL_dist_threshold]
                #merging_candidates = pd.unique(a[['c1', 'c2']].values.ravel('K'))
                #print(f"df cols: {df.columns}")
                df_ = df.copy()
                df_['original_label'] = df_[label_column]

                # TODO: merge all pairs within the specified threshold
                merged_classes = []
                classes_remap = {}
                num_superclasses = 0
                for i, row in df_class_separability.iterrows():
                    
                    if row['c1'] not in classes_remap.keys() and row['c2'] not in classes_remap.keys():
                        # Neither c1 nor c2 were merged into a superclass yet. Then, merge them.
                        num_superclasses +=1
                        # We use the label of c1 to the superclass
                        classes_remap[row['c1']] = row['c1'] #f"S{num_SuperClasses}"
                        classes_remap[row['c2']] = row['c1'] #f"S{num_SuperClasses}"

                    elif row['c1'] in classes_remap.keys() and row['c2'] in classes_remap.keys():
                        # Both c1 and c2 were already merged into a superclass
                        # Ex.: class C (merged in [B, C]) is close to D (in [D, E]). Then, merge both superclasses.
                        for k, v in classes_remap.items():
                            if v == classes_remap[row['c2']]:
                                classes_remap[k] = classes_remap[row['c1']]

                    elif row['c1'] in classes_remap.keys():
                        # Only c1 is already merged. Then, add c2 to the same superclass
                        classes_remap[row['c2']] = classes_remap[row['c1']]

                    elif row['c2'] in classes_remap.keys():
                        # Only c2 is already merged. Then, add c1 to the same superclass
                        classes_remap[row['c1']] = classes_remap[row['c2']]

                    #if row['c1'] not in merged_classes and row['c2'] not in merged_classes:
                        # merge c1 and c2
                        # NOTE: The superclass use the class label of c1
                        #df_.loc[df_[label_column]==row['c2'], label_column] = row['c1']
                
                    #merged_classes.append(row['c1'])
                    #merged_classes.append(row['c2'])
                    #self.superclasses.append((row['c1'], row['c2']))        # ALTERAR DEFINIÇÃO DE SUPERCLASSE. Pode ter mais de 2 classes.

                def group_by_superclass(d):
                    result = {}
                    for k, v in d.items():
                        result.setdefault(v, []).append(k)

                    ordered_result = {}
                    for k, v in result.items():
                        ordered_result[min(v)] = v
                    
                    return ordered_result

                self.superclasses = group_by_superclass(classes_remap)
                
                for superclass_label, original_labels in self.superclasses.items():
                    df_.loc[df_[label_column].isin(original_labels), label_column] = superclass_label #f"S{num_SuperClasses}"

                print("Superclasses: ", self.superclasses)
                print(f"df_ cols: {df_.columns}")

                AGL_unique_labels = np.unique(df_[label_column])
                self.AGL_n_classes = len(AGL_unique_labels)

                # TODO: initial classification, including superclasses
                #----
                print("Performing initial classification...")
                start = time.process_time()
                if self.AGL_n_classes > 2:
                    if AGL_default_strategy == 'OAO':
                        #self.OAO_strategy(df_.copy(), label_column, figsize, aspect, exclude_cols=['original_label'])
                        self.OAO_strategy(df_, label_column, figsize, aspect, exclude_cols=['original_label'])
                    elif AGL_default_strategy == 'OAA':
                        #self.OAA_strategy(df_.copy(), label_column, figsize, aspect, exclude_cols=['original_label'])
                        self.OAA_strategy(df_, label_column, figsize, aspect, exclude_cols=['original_label'])
                    else:
                        raise ValueError(
                            "Default AGL strategy not found.\n"
                            + "The standard multiclass classification strategies implemented include: OAO and OAA"
                        )
                elif self.AGL_n_classes == 2:
                    self.binary_svm(df_.copy(), label_column, figsize, aspect, exclude_cols=['original_label'])
                else:
                    # All initial classes were merged into a unique superclass
                    pass
                
                self.time.append({'function': 'initil classification', 'time': time.process_time() - start, 'msg': f"OAO, OAA or binary SVM"})
                #----
                """
                - iterate over self.superclasses
                    - if at least one vector was assigned to this superclass in the initial classification
                        - get their original dataframe indices
                        - filter the original dataframe to include only the classes that compose the superclass
                        - fit multiple classifiers, testing distinct parameters and strategies to optimize the classification of this superclass
                            - compute accuracy measures for each classifier, using the available test samples
                        - select the classifier that produced the most accurate result
                        - reclassify these vectors using the selected classifier
                            - change their classification output with the new labels (use their df indices)
                """
                
                # redefine the label values
                df_[label_column] = df_['original_label']
                df_.drop(['original_label'], axis=1, inplace=True)

                if list_kernels is None:
                    list_kernels = ['linear', 'rbf', 'polynomial', 'sigmoid']

                if list_C is None:
                    #list_C = np.arange(start=0, stop=1.2, step=0.2) #arange: intervalo aberto no final
                    #list_C = np.array([0.2, 0.3, 0.4, 1, 10, 100])
                    list_C = np.concatenate((np.arange(0, 1, 0.1), 
                                            np.arange(1, 5.5, 0.5), np.array([10])))

                if list_gammas is None:
                    #list_gammas = np.arange(start=0, stop=1.2, step=0.2) #arange: intervalo aberto no final
                    list_gammas = np.array([1, 0.8, 0.5, 0.2, 0.1])
                    # Insert 'scale' and 'auto' gamma values
                    # Scale: 1 / (n_features * X.var())
                    scale_gamma = 1/(self.dim * self.X.var())
                    # Auto: 1 / n_features
                    auto_gamma = 1/self.dim
                    list_gammas = np.insert(list_gammas, 0, auto_gamma)
                    list_gammas = np.insert(list_gammas, 0, scale_gamma)

                if list_degrees is None:
                    list_degrees = np.arange(start=1, stop=5, step=1) #arange: intervalo aberto no final


                # TODO: identify the best parameters to classify each superclass
                #superclass_classifiers = OrderedDict()
                #for c1, c2 in self.superclasses:
                for superclass_label, original_labels in self.superclasses.items():
                    
                    for i, (c1, c2) in enumerate(list(itertools.combinations(original_labels, 2) )):
                        print(f"Fitting parameters for superclass ({c1}, {c2})")
                    
                        clf_id = 0
                        clfs = OrderedDict()
                        stop = False
                        for k in list_kernels:
                            if stop:
                                break
                            
                            if k == 'rbf':
                                params = ['gamma']
                                extra_params = list_gammas
                            elif k == 'sigmoid':
                                params = ['gamma']
                                extra_params = list_gammas
                            elif k == 'polynomial':
                                params = ['degree']
                                extra_params = list_degrees
                            else:
                                params = [None]
                                extra_params = [None]

                            for j, (C, param) in enumerate(itertools.product(list_C, extra_params) ):
                                start = time.process_time()
                        
                                clfs[clf_id] = {"clf_id":clf_id, "k":k, "C":C}

                                params_dict = {}
                                for p_name in params:
                                    if p_name is not None:
                                        clfs[clf_id][p_name] = param
                                        params_dict[p_name] = param

                                #if self.verbose:
                                    #print(f"clf_id: {clf_id}, kernel: {k}, C: {C}, extra_params: {params_dict}")
                                    #print('params_dict')
                                    #print(params_dict)

                                df_superclass = df_.copy()
                                #indices_df = df_.index[df[label_column].isin([c1, c2])].tolist()
                                
                                df_superclass = df_superclass.loc[df_superclass[label_column].isin([c1, c2])]
                                #df_superclass = df_superclass.loc[df_superclass[label_column].isin([c1, c2])]
                                
                                df_validation = validation_set.loc[validation_set[label_column].isin([c1, c2])].copy()
                                #df_validation = validation_set.loc[validation_set[label_column].isin([c1, c2])].copy()

                                try:
                                    # REVIEW: strategy required if the superclass represent more than two classes
                                    clf = SVM(kernel=k, C=C, colors=self.colors, verbose=False, 
                                                minx=self.minx, maxx=self.maxx, 
                                                miny=self.miny, maxy=self.maxy, **params_dict)
                                    clf.params = params_dict
                                    
                                    clf.fit(df_superclass.copy(), label_column=label_column, aspect='auto')

                                    clfs[clf_id]['clf'] = clf
                                    clfs[clf_id]['n_svs'] = len(np.concatenate(clf.sv))

                                    for vr, row in df_validation.drop(label_column, axis=1).iterrows():
                                        #print(row.to_numpy())
                                        d, class_label = clf.predict(row.to_numpy())
                                        
                                        df_validation.loc[vr, 'AGL_label'] = class_label

                                    # TODO: compute accuracy statistics
                                    # REVIEW: Can I compute the accuracy statistics using the same samples used to produce the hyperplanes?
                                    # REVIEW: We could test these multiple classifiers in the classification function
                                    #--
                                    predicted_arr = df_validation['AGL_label'].to_numpy()
                                    expected_arr = df_validation[label_column].to_numpy()

                                    OA = round(accuracy_score(expected_arr, predicted_arr), 4)
                                    kappa_score = round(cohen_kappa_score(expected_arr, predicted_arr), 3)
                                    
                                    clfs[clf_id]['OA'] = OA
                                    clfs[clf_id]['kappa'] = kappa_score

                                    df_validation.drop('AGL_label', axis=1, inplace=True)

                                except Exception as err:
                                    print("Error: ", err)
                                    clfs[clf_id]['clf'] = None
                                    OA = 0
                                    clfs[clf_id]['OA'] = OA
                                    clfs[clf_id]['kappa'] = 0

                                clf_id = clf_id + 1
                                self.time.append({'function': f'fit: Fitting parameters {C}', 'time': time.process_time() - start, 'msg': f'fit: Fitting parameters for superclass ({c1}, {c2})'})

                                if OA >= self.tunning_OA_threshold:
                                    """
                                    There is no reason to test new classification setups if it was already found
                                    one that provides a satisfactory separation of the training samples in the 
                                    available feature space.
                                    """
                                    stop = True
                                    break

                        df_clfs = pd.DataFrame.from_dict(clfs, "index")
                        df_clfs.sort_values(by=['OA', 'kappa', 'n_svs'], ascending=[False, False, True], inplace=True)
                        
                        self.superclass_tests.append(clfs)
                        self.superclass_tests_summary.append(df_clfs)
                        clf = df_clfs.iloc[0]['clf']

                        self.superclass_classifiers.append(clf) # Chosen classifier
                        self.superclass_per_classifier.append(superclass_label) # Superclass that clf was chosen to classify
                        
                        # TODO: need to analyse the strategy (OAO, OAA) if the superclass include more than 2 classes
                        df_superclass.loc[df_superclass[label_column]==c1, 'temp_label'] = +1
                        df_superclass.loc[df_superclass[label_column]==c2, 'temp_label'] = -1

                        ## data
                        exclude_cols = [label_column, 'temp_label', 'original_label']
                        x = df_superclass.loc[:, ~df_superclass.columns.isin(exclude_cols)].to_numpy()
                        # labels: only the last column
                        y = df_superclass.loc[:, 'temp_label'].to_numpy().ravel()

                        clf.verbose = True
                        #self.binary_fit(x, y)

                        if clf.dim == 2:
                            clf.plot_margin(0, x[y == 1], x[y == -1], c1, c2, figsize=figsize, aspect=aspect)
                        else:
                            print(f"Plot não exibido. Dim: {clf.dim}. Cols: {clf.X_cols}")

            else:
                raise ValueError(
                    "Strategy not found.\n"
                    + "The implemented multiclass classification strategies are: OAO, OAA and AGL"
                )
        elif n_classes == 2:
            # two-class classification problem
            self.binary_svm(df.copy(), label_column, figsize, aspect, exclude_cols=['original_label'])

            #c1 = unique_labels[0]
            #c2 = unique_labels[1]
            #if self.verbose:
            #    print(f"Two-class classification problem: class {c1} x class {c2}")
            #
            #self.c1.append(c1)
            #self.c2.append(c2)
            #
            #df.loc[df[label_column]==c1, 'temp_label'] = +1
            #df.loc[df[label_column]==c2, 'temp_label'] = -1
            #
            ### data
            ## feature values: all columns but the last
            ##x = df.iloc[:,:-1].to_numpy()
            #x = df.loc[:, ~df.columns.isin([label_column, 'temp_label'])].to_numpy()
            ## labels: only the last column
            ##y = df.iloc[:,-1].to_numpy()
            #y = df.loc[:, df.columns == 'temp_label'].to_numpy().ravel()
            #
            #self.binary_fit(x, y)
            #
            #if self.dim == 2:
            #    self.plot_margin(0, x[y == 1], x[y == -1], c1, c2, figsize=figsize, aspect=aspect)
            
        else:
            raise ValueError(
                "The dataset should contain at least 2 classes."
            )


    def classify(self, dataset, label_column=None, test_dataset=None, test_label_column=None, test_expected_arr=None):

        def AGL_classification(self, data_arr):
            # Predicted labels - considering superclasses, if any
            predicted_list, _ = self.predict(data_arr)
            
            # Reclassify pixels assigned to superclasses
            predicted_arr = np.array(predicted_list)
            
            #for i, (superclass_label, original_labels) in enumerate(self.superclasses.items()):
            ##for i, (c1, c2) in enumerate(self.superclasses):
            #    start = time.process_time()
            #
            #    clf = self.superclass_classifiers[i]
            #    
            #    SC_indices = np.where(predicted_arr == superclass_label)[0]
            #    X_temp = data_arr[SC_indices]
            #    predicted_temp, _ = clf.predict(X_temp)
            #    predicted_arr[SC_indices] = predicted_temp
            #
            #    self.time.append({'function': 'AGL_classification', 'time': time.process_time() - start, 'msg': f"Class {superclass_label}"})

            for i, (superclass_label, original_labels) in enumerate(self.superclasses.items()):
                start = time.process_time()

                SC_indices = np.where(predicted_arr == superclass_label)[0]
                X_temp = data_arr[SC_indices]

                SC_clfs_indices = np.where(np.array(self.superclass_per_classifier) == superclass_label)
                clfs = np.array(self.superclass_classifiers)[SC_clfs_indices]

                predicted_temp, _ = self.predict(X_temp, AGG=True, clfs=clfs)
                predicted_arr[SC_indices] = predicted_temp

                self.time.append({'function': 'AGL_classification', 'time': time.process_time() - start, 'msg': f"Class {superclass_label}"})
            ##predicted_list = list(predicted_arr)
            
            return predicted_arr #predicted_list

        def accuracy_assessment(expected_arr, predicted_arr):
            OA_test = round(accuracy_score(expected_arr, predicted_arr), 4)
            kappa_test = round(cohen_kappa_score(expected_arr, predicted_arr), 3)

            # confusion matrix for the testing dataset
            confusion_pred = pd.crosstab(expected_arr, predicted_arr, rownames=['Actual'], colnames=['Predicted'], margins=True, margins_name="Total")
            expected_rows = list(pd.unique(expected_arr))
            expected_rows.append('Total')
            
            confusion_pred = confusion_pred.reindex(index=expected_rows, columns=expected_rows, fill_value=0) #Required in case a antagonistic class is not found in 'predicted_test'

            for c in np.unique(expected_arr):
                confusion_pred.loc[c, 'PA']= str(round(confusion_pred.loc[c, c]/confusion_pred.loc[c, "Total"], 3))
                confusion_pred.loc['UA', c]= str(round(confusion_pred.loc[c, c]/confusion_pred.loc["Total", c], 3))

            #confusion_pred.loc['OA (resubstitution)',confusion_pred.columns[0]] = str(round(OA_train, 3))
            confusion_pred.loc['OA (Holdout)',confusion_pred.columns[0]] = str(round(OA_test, 3))                
            confusion_pred.loc['Kappa',confusion_pred.columns[0]] = str(round(kappa_test, 3)) 

            # Renaming the classes (including the counter-example names to avoid misinterpretation)
            cols_names = confusion_pred.columns.values
            rows_names = confusion_pred.index.values
            #for key in dict_CE:
            #    cols_names[cols_names==key] = dict_CE[key]
            #    rows_names[rows_names==key] = dict_CE[key]
            
            confusion_pred.columns = pd.Index(cols_names, name='Predicted')
            confusion_pred.index = pd.Index(rows_names, name='Actual')

            return OA_test, kappa_test, confusion_pred


        if type(dataset) == pd.DataFrame:
            if label_column is None:
                data_arr = dataset.to_numpy()

            elif label_column in dataset.columns:
                data_arr = dataset.drop(label_column, axis=1)
                data_arr = data_arr.to_numpy()
            
            else: #label_column is not None and label_column not in dataset.columns:
                raise ValueError(
                    f"""Column of labels not identified in the dataframe.
                    The column {label_column} does not exist in the dataset."""
                )
        
        elif type(dataset == np.ndarray):
            data_arr = dataset.copy()
        
        else:
            raise ValueError(
                "Dataset type not accepted."
                + "The dataset parameter should be a numpy array or a dataframe."
            )

        if test_dataset is None:
            OA_test = None
            kappa_test = None
            confusion_pred = None

        elif type(test_dataset) == pd.DataFrame:
            if test_label_column is None:
                if test_expected_arr is not None and (type(test_expected_arr)==np.array or type(test_expected_arr)==list):
                    test_arr = test_dataset.to_numpy()
                    if type(test_expected_arr) == list:
                        test_expected_arr = np.array(test_expected_arr)
                else:
                    raise ValueError(
                            "Expected labels required for the test dataset."
                            + "The test_expected_arr parameter cannot be None if the label column is not provided for the test dataset."
                        )

            elif test_label_column in test_dataset.columns:
                test_arr = test_dataset.drop(test_label_column, axis=1)
                test_arr = test_arr.to_numpy()
                
                test_expected_arr = test_dataset[test_label_column].to_numpy()
            
            elif test_expected_arr is not None and (type(test_expected_arr)==np.array or type(test_expected_arr)==list):
                test_arr = test_dataset.to_numpy()
                if type(test_expected_arr) == list:
                    test_expected_arr = np.array(test_expected_arr)
            
            else: #label_column is not None and label_column not in dataset.columns:
                raise ValueError(
                    f"""Column of labels not identified in the test dataframe and test_expected_arr not provided.
                    The column {test_label_column} does not exist in the test dataset and the test_expected_arr parameter was not provided (list or np.array)."""
                )
        
        elif type(test_dataset == np.ndarray):
            if test_expected_arr is not None and (type(test_expected_arr)==np.array or type(test_expected_arr)==list):
                test_arr = test_dataset.copy()
                if type(test_expected_arr) == list:
                    test_expected_arr = np.array(test_expected_arr)
            else:
                raise ValueError(
                    "Expected labeld not provided for the test dataset."
                    + "The test_expected_arr parameter must be provided for test dataset of type numpy array."
                )
        
        else:
            raise ValueError(
                "Dataset type not accepted for the test data."
                + "The test_dataset parameter should be a numpy array or a dataframe."
            )

        predicted_arr = AGL_classification(self, data_arr)

        if test_dataset is not None:
            test_predicted_arr = AGL_classification(self, test_arr)
            OA_test, kappa_test, confusion_pred = accuracy_assessment(test_expected_arr, test_predicted_arr)

        accuracy_dict = {'OA': OA_test, 'Kappa': kappa_test, 'confusion_matrix': confusion_pred}
        
        #if test_mask_arr is not None:
        #    
        #    for c, class_ in self.classes:
        #        indexes = np.where(test_mask_arr==c)
        #        if (indexes[0].size > 0):

        return predicted_arr, accuracy_dict

        
    def class_separability_analysis(self, method='JM'):

        start = time.process_time()

        d = OrderedDict()
        for i, (c1, c2) in enumerate(list(itertools.combinations(self.classes, 2) )):
            dist = self.distances(c1=c1, c2=c2, method=method)

            d[i] = {"pair":i, "c1":c1, "c2":c2, "distance":dist}

        self.separability_method = method

        df_class_separability = pd.DataFrame.from_dict(d, "index")
        df_class_separability.sort_values(by=['distance'], ascending=True, inplace=True)
        
        self.class_separability = df_class_separability

        self.time.append({'function': 'class_separability_analysis', 'time': time.process_time() - start, 'msg': None})

        return self.class_separability


    # Stochastic distances
    def distances(self, c1, c2, method='JM'):
        # Richards (2013, p. 350)
        # https://github.com/KolesovDmitry/i.jmdist/blob/master/i.jmdist
        # https://jneuroengrehab.biomedcentral.com/articles/10.1186/s12984-017-0283-5/tables/2

        if not hasattr(self, 'x') or not hasattr(self, 'y'):
            raise ValueError(
                "x and y not defined in object.\n"
                + "Fit the SVM model before computing the separability distances."
            )

        if c1 not in self.classes or c2 not in self.classes:
            raise ValueError(
                "Both c1 and c2 must be amongst the object classes."
            )

        def mahalanobis_distance(m1, m2, cov1, cov2):
            m = (m1 - m2)
            cov = (cov1 + cov2)/2
            inv_cov = np.linalg.inv(cov)

            tmp = np.core.dot(m.T, inv_cov)
            tmp = np.core.dot(tmp, m)
            
            MH = np.sqrt(tmp)
            return MH


        def bhattacharyya_distance(m1, m2, cov1, cov2):
            
            m = (m1 - m2)
            cov = (cov1 + cov2)/2
            inv_cov = np.linalg.inv(cov)

            pt1 = np.core.dot(m.T, inv_cov)
            pt1 = np.core.dot(pt1, m)
            pt1 = pt1/8.0
            
            pt2 = np.linalg.det(cov) / np.sqrt( np.linalg.det(cov1)*np.linalg.det(cov2) )
            pt2 = np.log(pt2)
            pt2 = pt2/2.0

            B = pt1 + pt2
            return B


        def jeffries_matusita_distance(m1, m2, cov1, cov2):
            # JM = 2(1-np.e(-B)) :: for normally distributed classes
            B = bhattacharyya_distance(m1, m2, cov1, cov2)

            #JM = np.sqrt(2 * (1 - np.exp(-B)))
            JM = 2 * (1 - np.exp(-B))
            return JM

        id1 = np.where(self.classes == c1)[0][0]
        id2 = np.where(self.classes == c2)[0][0]

        if method == 'Bhattacharyya' or method == 'B':
            dist = bhattacharyya_distance(self.class_mean[id1], self.class_mean[id2], 
                                            self.class_cov[id1], self.class_cov[id2])

        elif method == 'Jeffries-Matusita' or method == 'JM':
            dist = jeffries_matusita_distance(self.class_mean[id1], self.class_mean[id2], 
                                                self.class_cov[id1], self.class_cov[id2])

        elif method == 'Mahalanobis' or method == 'M':
            dist = mahalanobis_distance(self.class_mean[id1], self.class_mean[id2], 
                                            self.class_cov[id1], self.class_cov[id2])
        else:
            raise ValueError(
                    "Method not found.\n"
                    + "The implemented stochastic distances are: Bhattacharyya; Jeffries-Matusita (JM); and, Mahalanobis."
                )
        
        return dist


    def add_point(self, ax, x1, x2, s=50, color=[0.3, 0.3, 0.3], label=None):
        ax.scatter(x1, x2, s=s, color=color, label=label)
        return ax


    def plot_multiclass_boundary(self, point=None, point_label=None, all_margins=False, 
                                    figsize=(12,5), aspect='equal'):
        """
        # NOTE: REFERENCE
        - based on the mlxtend library
        https://github.com/rasbt/mlxtend/blob/master/mlxtend/plotting/decision_regions.py
        https://github.com/rasbt/mlxtend/blob/8c61c063f98f0d9646edfd8b0270b77916f0c434/mlxtend/utils/checking.py
        """
        
        def plot_fig(self, xx, yy, Z, colors, contourf_kwargs, contour_kwargs, 
                        n_classes, all_margins, point=None, point_label=None,
                        figsize=(12,5), aspect='equal'):
            
            # initialize figure
            fig = plt.figure(figsize=figsize)

            gs = gridspec.GridSpec(1, 3, width_ratios=[1,3,1]) 

            # setup current axis
            #aspect = {'auto', 'equal'} or float
            ax = plt.subplot(gs[1], aspect=aspect) 

            cset = ax.contourf(xx, yy, Z,
                            colors=colors,
                            levels=np.arange(Z.max() + 2) - 0.5,
                            **contourf_kwargs)

            if all_margins is True:
                for margin in self.margins:
                    if self.kernel  != 'linear':
                        ax.contour(margin[0], margin[1], margin[2], [0.0], colors='k', linewidths=2.5, origin='lower')
                    else:
                        ax.plot([self.minx, self.maxx], [margin[0], margin[1]], "k", linewidth=2.5)
                    
                    #for i, (c1, c2) in enumerate(self.superclasses):
                    for clf in self.superclass_classifiers:
                        #clf = self.superclass_classifiers[i]
                        for margin in clf.margins:
                            if clf.kernel  != 'linear':
                                ax.contour(margin[0], margin[1], margin[2], [0.0], colors='k', linewidths=2.5, origin='lower')
                            else:
                                ax.plot([clf.minx, clf.maxx], [margin[0], margin[1]], "k")
            else:
                ax.contour(xx, yy, Z, cset.levels,
                            **contour_kwargs)

            ax.axis([xx.min(), xx.max(), yy.min(), yy.max()])

            #ax = self.add_point(ax, x1=0.4, x2=0.435, label='Point')
            
            #for a in range(0, num_classes):
            for a in list(np.unique(self.Y)):
                t = np.argwhere(self.Y == a)
                t = t[:,0]
                ax.scatter(self.X[t,self.f1], self.X[t,self.f2], s=50, color=self.colors[int(a)], label=f'Class {a}')#, edgecolor='k', linewidth=1.5)
                
            if point is not None:
                if point_label is None:
                    point_label="Pixel"
                #ax.plot(point[0], point[1], marker='x', markersize=8, color='black')
                ax.scatter(point[0], point[1], s=50, color='black', label=point_label)

            # Legend
            if point is not None:
                ncol = n_classes + 1
            else:
                ncol = n_classes

            # Legend
            fig.legend(loc='lower center', fancybox=True, shadow=False, ncol=ncol, 
                                title_fontsize=8, framealpha=0, labelspacing=1, edgecolor='k')#, title="Legend")

        if self.dim > 2:
            print("SVM model fitted to more than two features. 2D plots require a change of dimensionality to correctly illustrate the decision boundaries.")
            self.show_dataset()
        else:
            if self.xx is None or self.yy is None or self.Z is None:
                x_min = self.minx
                x_max = self.maxx
                y_min = self.miny
                y_max = self.maxy
                dim = self.dim

                xnum, ynum = plt.gcf().dpi * plt.gcf().get_size_inches()
                xnum, ynum = math.floor(xnum), math.ceil(ynum)

                # meshgrid
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, num=xnum),
                                    np.linspace(y_min, y_max, num=ynum))
                self.xx = xx
                self.yy = yy

                # Prediction array
                if dim == 1:
                    X_predict = np.array([xx.ravel()]).T
                else:
                    X_grid = np.array([xx.ravel(), yy.ravel()]).T
                    #X_predict = np.zeros((X_grid.shape[0], dim))
                    X_predict = np.zeros((X_grid.shape[0], 2))
                    X_predict[:, 0] = X_grid[:, 0]
                    X_predict[:, 1] = X_grid[:, 1]

                # Predicted labels
                Z_, _ = self.predict(X_predict.astype(self.X.dtype))
                
                # Reclassify pixels assigned to superclasses
                Z_ = np.array(Z_)
                self.Z_initial = Z_

                # #for i, (superclass_label, original_labels) in enumerate(self.superclasses.items()):
                # for clf in self.superclass_classifiers:
                # #for i, (c1, c2) in enumerate(self.superclasses):

                #     #clf = self.superclass_classifiers[i]
                    
                #     SC_indices = np.where(Z_ == superclass_label)[0]
                #     X_temp = X_predict[SC_indices]
                #     Z_temp, _ = clf.predict(X_temp)
                #     Z_[SC_indices] = Z_temp
                
                for i, (superclass_label, original_labels) in enumerate(self.superclasses.items()):
                    SC_indices = np.where(Z_ == superclass_label)[0]
                    X_temp = X_predict[SC_indices]

                    SC_clfs_indices = np.where(np.array(self.superclass_per_classifier) == superclass_label)
                    clfs = np.array(self.superclass_classifiers)[SC_clfs_indices]

                    Z_temp, _ = self.predict(X_temp, AGG=True, clfs=clfs)
                    Z_[SC_indices] = Z_temp
                
                self.Z_final = Z_
                Z_ = list(Z_)

                # TODO: only for test
                #Z_ = pd.read_csv('Z_.csv')
                #Z_ = list(Z_.to_numpy().ravel())
                
                self.Z_ = Z_
                Z = np.array(Z_).reshape(xx.shape)
                self.Z = Z
            else:
                xx = self.xx
                yy = self.yy
                Z = self.Z


            contourf_kwargs = None#{'extend': 'both', 'corner_mask': True}
            contour_kwargs = {'alpha': 0.9, 'antialiased': True, 'linestyles':'solid',
                                'linewidths':3, 'extend': 'both'}

            n_classes = np.unique(self.Y).shape[0]
            #colors=('#1f77b4,#ff7f0e,#3ca02c,#d62728,'
            #    '#9467bd,#8c564b,#e377c2,'
            #    '#7f7f7f,#bcbd22,#17becf')
            #colors = colors.split(',')
            colors_gen = cycle(self.colors)
            colors = [next(colors_gen) for c in range(n_classes)]

            # Plot decisoin region
            # Make sure contourf_kwargs has backwards compatible defaults
            contourf_kwargs_default = {'alpha': 0.25, 'antialiased': True}
            contourf_kwargs = format_kwarg_dictionaries(
                                default_kwargs=contourf_kwargs_default,
                                user_kwargs=contourf_kwargs,
                                protected_keys=['colors', 'levels'])

            contour_kwargs_default = {
                'linewidths': 0.5, 'colors': 'k', 'antialiased': True}
            contour_kwargs = format_kwarg_dictionaries(
                                default_kwargs=contour_kwargs_default,
                                user_kwargs=contour_kwargs,
                                protected_keys=[])
            
            if all_margins is True:
                plot_fig(self, xx, yy, Z, colors, contourf_kwargs, contour_kwargs, 
                            n_classes, all_margins=all_margins, point=point, point_label=point_label,
                            figsize=figsize, aspect=aspect)
            
            plot_fig(self, xx, yy, Z, colors, contourf_kwargs, contour_kwargs, 
                            n_classes, all_margins=False, point=point, point_label=point_label,
                            figsize=figsize, aspect=aspect)


    def show_dataset(self, multiclass_boundary=False, point=None, point_label=None,
                        figsize=(12,5), aspect='equal', f1=None, f2=None):
        
        if self.X is not None and self.Y is not None:
            # initialize figure
            fig = plt.figure(figsize=figsize)

            gs = gridspec.GridSpec(1, 3, width_ratios=[1,3,1]) 

            # setup current axis
            ax = plt.subplot(gs[1], aspect=aspect); 

            # dress panel
            ax.set_xlim(self.minx, self.maxx)
            ax.set_ylim(self.miny, self.maxy)

            num_classes = np.size(np.unique(self.Y))

            #cmap = LinearSegmentedColormap.from_list('my_list', colors[:num_classes])
            #scatter = plt.scatter(self.X[:, 0], self.X[:, 1], c=self.Y, s=50, cmap=cmap, edgecolor='k', linewidth=1.5)

            #fig.legend(handles=scatter.legend_elements()[0], labels=[f"Class {c}" for c in list(np.unique(self.Y))],
            #        loc='lower center', fancybox=True, shadow=False, ncol=len(list(np.unique(self.Y))), 
            #        title_fontsize = 9, framealpha=0, labelspacing=1, edgecolor='k')#, title = "Legend")

            # color current class
            #for a in range(0, num_classes):
            for a in list(np.unique(self.Y)):
                t = np.argwhere(self.Y == a)
                t = t[:,0]
                if f1 is not None and f2 is not None:
                    ax.scatter(self.X[t,f1], self.X[t,f2], s=50, color=self.colors[int(a)], label=f'Class {a}')#, edgecolor='k', linewidth=1.5)
                else:
                    ax.scatter(self.X[t,self.f1], self.X[t,self.f2], s=50, color=self.colors[int(a)], label=f'Class {a}')#, edgecolor='k', linewidth=1.5)

            
            # Plot support vectors
            ##ax.scatter(self.sv[:, 0], self.sv[:, 1], s=50, linewidth=1.2, facecolors="none", edgecolor="k")#, label="Support vectors")
            #ax.scatter(self.x[self.sv_bool, 0], self.x[self.sv_bool, 1], s=50, linewidth=1.5, facecolors="none", edgecolor="k")#, label="Support vectors")

            if point is not None:
                #ax.plot(point[0], point[1], marker='x', markersize=8, color='black')
                ax.scatter(point[0], point[1], s=50, color='black', label=f'Pixel')

            if multiclass_boundary is True:
                if self.dim > 2:
                    print("SVM model fitted to more than two features. 2D plots require a change of dimensionality to correctly illustrate the decision boundaries.")
                
                if (f1 is None or f2 is None) or (f1==self.f1 and f2==self.f2):

                    for margin in self.margins:
                        if self.kernel  != 'linear':
                            ax.contour(margin[0], margin[1], margin[2], [0.0], colors='k', linewidths=2.5, origin='lower')
                        else:
                            ax.plot([self.minx, self.maxx], [margin[0], margin[1]], "k")
                    
                    #for i, (c1, c2) in enumerate(self.superclasses):
                    for clf in self.superclass_classifiers:
                        #clf = self.superclass_classifiers[i]
                        for margin in clf.margins:
                            if clf.kernel  != 'linear':
                                ax.contour(margin[0], margin[1], margin[2], [0.0], colors='k', linewidths=2.5, origin='lower')
                            else:
                                ax.plot([clf.minx, clf.maxx], [margin[0], margin[1]], "k")

            # Legend
            if point is not None:
                ncol = num_classes + 1
            else:
                ncol = num_classes
            fig.legend(loc='lower center', fancybox=True, shadow=False, ncol=ncol, 
                    title_fontsize=8, framealpha=0, labelspacing=1, edgecolor='k')#, title="Legend")

            plt.show()


    def project_features(self, V, id):

        # In plot functions only 2 features are considered, even if the dataset contains multiple features
        n_features = V.shape[1]

        # Create the decision boundary for the plots. Calculates the hypothesis.
        if self.w[id] is not None:
            if n_features == 2:
                w = self.w[id][[self.f1, self.f2]]
            else:
                w = self.w[id]
            return np.dot(V, w) + self.b[id]
        else:
            y_predict = np.zeros(len(V))
            for i in range(len(V)):
                s = 0
                for a, sv_y, sv in zip(self.alphas[id], self.sv_y[id], self.sv[id]):
                    # a : Lagrange multipliers, sv : support vectors.
                    # Hypothesis: sign(sum^S a * y * kernel + b)

                    if n_features == 2:
                        sv = sv[[self.f1, self.f2]]

                    if self.kernel == 'linear':
                        s += a * sv_y * self.linear_kernel(V[i], sv)

                    elif self.kernel=='rbf':
                        s += a * sv_y * self.rbf_kernel(V[i], sv, self.gamma)   # Kernel trick.
                        #self.C = None   # Not used in rbf kernel.
                    
                    elif self.kernel=='sigmoid':
                        s += a * sv_y * self.sigmoidal_kernel(V[i], sv, self.gamma, self.coef0)   # Kernel trick.

                    elif self.kernel == 'polynomial':
                        s += a * sv_y * self.polynomial_kernel(V[i], sv, self.gamma, self.coef0, self.degree)

                    else:
                        raise ValueError(f"Kernel function not found for '{self.kernel}'")

                y_predict[i] = s
                
            return y_predict + self.b[id]


    def project(self, V, id):
        
        # In plot functions only 2 features are considered, even if the dataset contains multiple features
        n_features = len(V)
    
        # Create the decision boundary for the plots. Calculates the hypothesis.
        if self.w[id] is not None:
            if n_features == 2:
                w = self.w[id][[self.f1, self.f2]]
            else:
                w = self.w[id]
            return np.dot(V, w) + self.b[id]
        else:
            #KERNEL SVM
            
            #y_predict = np.zeros(len(V))
            #for i in range(len(V)):
            #--
            s = 0
            for a, sv_y, sv in zip(self.alphas[id], self.sv_y[id], self.sv[id]):
                # a : Lagrange multipliers, sv : support vectors.
                # Hypothesis: sign(sum^S a * y * kernel + b)

                if n_features == 2:
                    sv = sv[[self.f1, self.f2]]

                if self.kernel == 'linear':
                    #s += a * sv_y * self.linear_kernel(V[i], sv)
                    s += a * sv_y * self.linear_kernel(V, sv)
                elif self.kernel=='rbf':
                    #s += a * sv_y * self.gaussian_kernel(V[i], sv, self.gamma)   # Kernel trick.
                    s += a * sv_y * self.rbf_kernel(V, sv, self.gamma)   # Kernel trick.
                    #self.C = None   # Not used in rbf kernel.
                elif self.kernel=='sigmoid':
                    s += a * sv_y * self.sigmoidal_kernel(V, sv, self.gamma, self.coef0)   # Kernel trick.
                elif self.kernel == 'polynomial':
                    #s += a * sv_y * self.polynomial_kernel(V[i], sv, self.C, self.degree)
                    s += a * sv_y * self.polynomial_kernel(V, sv, self.gamma, self.coef0, self.degree)
                else:
                    raise ValueError(f"Kernel function not found for '{self.kernel}'")

            #y_predict[i] = s
            y_predict = s
            #--
            return y_predict + self.b[id]


    def predict(self, V, AGG=False, clfs=None):
        
        def f(self, V, AGG):
            #df = pd.DataFrame([], columns=['classifier', 'c1', 'c2', 'fx', 'signal'])

            # for each binary classification
            d = OrderedDict()
            if AGG is False: 
                # Conventional approach: a unique classifier with multiple decision boundaries
                for id, (c1, c2) in enumerate(zip(self.c1, self.c2)):
                    fx = self.project(V, id)
                    
                    #if self.kernel != "linear":
                    #    fx = np.min(fx)

                    # NOTE: The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0.
                    sign = np.sign(fx)
                    
                    d[id] = {"classifier":id, "c1":c1, "c2":c2, "fx":fx, "signal":sign, "fx_c2":-fx, "signal_c2":-sign}
                    #df = df.append(d_id, ignore_index=True)
            else: 
                # Agglomerative approach: a distinct model to classify each pair of class
                for i, clf in enumerate(clfs): #self.superclass_classifiers:
                    id = 0
                    fx = clf.project(V, id)
                    
                    # NOTE: The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0.
                    sign = np.sign(fx)

                    d[i] = {"classifier":i, "c1":clf.c1[id], "c2":clf.c2[id], "fx":fx, "signal":sign, "fx_c2":-fx, "signal_c2":-sign}

            df = pd.DataFrame.from_dict(d, "index")
            
            # multi-class classification problem (more than 1 binary subproblem)
            if len(d) > 1:
                if self.strategy == 'OAA' or (self.strategy == 'AGL' and self.AGL_default_strategy == 'OAA'):
                    c = df.loc[:, 'c1'].to_numpy()
                    signal_labels = df.loc[:, 'signal'].to_numpy()
                    fx_values = df.loc[:, 'fx'].to_numpy()

                    index = np.argmax(signal_labels)
                    #print('index: ', index)
                    
                    # There is only one +1 label (no tie for the +1 label)
                    if signal_labels[index] == +1 and len(np.where(signal_labels==signal_labels[index])[0]) == 1:
                        class_label = c[index]

                    # There are no +1 labels or there is more than one class with the +1 label
                    else:
                        fx_index = np.argmax(fx_values)
                        class_label = c[fx_index]

                elif self.strategy == 'OAO' or (self.strategy == 'AGL' and self.AGL_default_strategy == 'OAO'):
                    c = df.loc[:, df.columns.isin(['c1', 'c2'])].to_numpy().ravel()
                    #df['signal_c2'] = df['signal'] * (-1)
                    signal_labels = df.loc[:, df.columns.isin(['signal', 'signal_c2'])].to_numpy().ravel()
                    
                    unique_c = np.unique(c)
                    sum_signals = [signal_labels[indices].sum() for x in unique_c for indices in np.where(c == x)]

                    index = np.argmax(sum_signals)

                    # There is at least one positive value and no tie for the argmax value
                    if sum_signals[index] > 0 and len(np.where(sum_signals==sum_signals[index])[0]) == 1:
                        class_label = unique_c[index]

                    # There are no positive labels or there is more than one class with the argmax value
                    else:
                        #df['fx_c2'] = df['fx'] * (-1)
                        fx_values = df.loc[:, df.columns.isin(['fx', 'fx_c2'])].to_numpy().ravel()
                        sum_fxs = [fx_values[indices].sum() for x in unique_c for indices in np.where(c == x)]

                        fx_index = np.argmax(sum_fxs)
                        class_label = c[fx_index]
                
                else:
                    raise ValueError(
                        "Strategy not found.\n"
                        + "The implemented multiclass classification strategies are: OAO, OAA and AGL"
                    )
            
            elif len(d)==1:
                # Two-class problem (only one binary subproblem)
                if df.loc[0, 'signal'] == +1:
                    class_label = df.loc[0, 'c1']
                else:
                    class_label = df.loc[0, 'c2']
            
            else:
                # If self.c1 is empty because there is only one class (the initial classification was not performed)
                if len(self.c1) == 0 and self.AGL_n_classes == 1 and len(self.superclasses) == 1:
                    superclass = list(self.superclasses.keys())[0]
                    class_label = superclass
                else:
                    raise ValueError(
                        "Empty prediction dataframe. No binary subproblems found."
                    )
            
            return df, class_label

        if V.ndim == 1:
            df, class_label = f(self, V, AGG)
            return df, class_label
        else:
            r = []
            for v in V:
                df, class_label = f(self, np.array(v), AGG)
                r.append(class_label)
            return r, None  


    def plot_margin(self, i, X1, X2, c1, c2=None, figsize=(12,5), aspect='equal'):
        #-------------------------------------------------------
        # Purpose: Plot the SVM data and margin.
        # Inputs:
        #       X1      :   Vector of X1 data.
        #       X2      :   Vector of X2 data.
        #-------------------------------------------------------

        def f(x, w, b, c=0, f1=0, f2=1):
            # -------------------------------------------------------
            # Purpose: Create the margin line given the intercept and coefficients (weights).
            #          Given X1, return X2 such that [X1,X2] in on the line:
            #                       w.X1 + b = c
            # Inputs:
            #       x       :       data
            #       w       :       Coefficient weights.
            #       c       :       Soft margin parameter. c=0 for hard margin.
            # -------------------------------------------------------
            #return (-w[0] * x - b + c) / w[1]
            return (-w[f1] * x - b + c) / w[f2]

        fig = plt.figure(figsize=figsize)  # create a figure object

        gs = gridspec.GridSpec(1, 3, width_ratios=[1,3,1]) 

        # setup current axis
        ax = plt.subplot(gs[1], aspect=aspect)
        
        ## determine plotting ranges
        #if self.X is not None:
        #    minx = min(min(self.X[:,0]), min(self.X[:,1]))
        #    maxx = max(max(self.X[:,0]), max(self.X[:,1]))
        #else:
        #    minx = min(X1.min(), X2.min())
        #    maxx = min(X1.max(), X2.max())
        #gapx = (maxx - minx) * 0.1
        #minx -= gapx
        #maxx += gapx

        # dress panel
        ax.set_xlim(self.minx, self.maxx)
        ax.set_ylim(self.miny, self.maxy)

        # Axis limits.
        #x1_min, x1_max = X1.min(), X1.max()
        #x2_min, x2_max = X2.min(), X2.max()
        #ax.set(xlim=(x1_min, x1_max), ylim=(x2_min, x2_max))

        for c, a in zip([c1, c2], [1, -1]):
            t = np.argwhere(self.y[i] == a)
            t = t[:,0]
            if c is None:
                c_name = 'Rest'
                c = -1
            else:
                c_name = c
                
            ax.scatter(self.x[i][t,self.f1], self.x[i][t,self.f2], s=50, color=self.colors[int(c)], label=f'Class {c_name}')#, edgecolor='k', linewidth=1.5)

        # Free support vectors
        #plt.scatter(self.free_sv[i][:, self.f1], self.free_sv[i][:, self.f2], s=90, linewidth=1.8, facecolors="none", edgecolor="dimgray")#, label="Free SV")

        # The points designating the support vectors.
        plt.scatter(self.sv[i][:, self.f1], self.sv[i][:, self.f2], s=70, linewidth=1.8, facecolors="none", edgecolor="k", label="Support vectors")

        if self.kernel  != 'linear':
            # Non-linear margin line needs to be generated. Will use a contour plot.
            _X1, _X2 = np.meshgrid(np.linspace(self.minx, self.maxx, 100), np.linspace(self.minx, self.maxx, 100))
            X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(_X1), np.ravel(_X2))])

            Z = self.project_features(X, i).reshape(_X1.shape)
            
            plt.contour(_X1, _X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
            self.margins.append((_X1, _X2, Z))

            plt.contour(_X1, _X2, Z + 1, [0.0], colors='grey', linestyles='--', linewidths=1, origin='lower')
            plt.contour(_X1, _X2, Z - 1, [0.0], colors='grey', linestyles='--', linewidths=1, origin='lower')
        else:
            # Linear margin line needs to be generated.
            # This section can be done by the above code and use plt.contour. But wanting to generate the linear lines here for demonstration.
            # Decision Boundary:  w.x + b = 0
            _y1 = f(self.minx, self.w[i], self.b[i], f1=self.f1, f2=self.f2)
            _y2 = f(self.maxx, self.w[i], self.b[i], f1=self.f1, f2=self.f2)
            plt.plot([self.minx, self.maxx], [_y1, _y2], "k")
            self.margins.append((_y1, _y2))

            # Margin Upper: w.x + b = 1
            _y3 = f(self.minx, self.w[i], self.b[i], 1, f1=self.f1, f2=self.f2)
            _y4 = f(self.maxx, self.w[i], self.b[i], 1, f1=self.f1, f2=self.f2)
            plt.plot([self.minx, self.maxx], [_y3, _y4], "k--")

            # Margin Lower: w.x + b = -1
            _y5 = f(self.minx, self.w[i], self.b[i], -1, f1=self.f1, f2=self.f2)
            _y6 = f(self.maxx, self.w[i], self.b[i], -1, f1=self.f1, f2=self.f2)
            plt.plot([self.minx, self.maxx], [_y5, _y6], "k-.")

        # Legend
        fig.legend(loc='lower center', fancybox=True, shadow=False, ncol=3,
                title_fontsize=9, framealpha=0, labelspacing=1, edgecolor='k')#, title="Legend")

        if self.verbose:
            plt.show()
        else:
            """Figures created through the pyplot interface (matplotlib.pyplot.figure) are retained 
            until explicitly closed and may consume too much memory.
            """
            plt.close(fig)