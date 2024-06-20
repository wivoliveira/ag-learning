import math
import time
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copy import deepcopy
from itertools import cycle
from sklearn.svm import SVC
from matplotlib import gridspec
from collections import OrderedDict
from sklearnex import patch_sklearn
from sklearn.multiclass import OneVsRestClassifier
from sklearn.experimental import enable_halving_search_cv # IMPORTAR ANTES DE IMPORTAR HalvingGridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, cohen_kappa_score
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, StratifiedKFold
patch_sklearn()

from .help_functions import format_kwarg_dictionaries
# from .class_separability import class_separability_analysis


class AGG_SVC:

    def __init__(self,
        strategy='AGG', kernel='linear', C=100, gamma='scale', degree=2, coef0=1, HAGG_N = None,
        AGG_classify_original = True, distance_restriction = False, scale_data=False, 
        interpolate_nan=False, distance_method='JM', AGG_dist_threshold=1.5,
        AGG_default_strategy = 'OAA',

        GridSearchCV = True, halving = True, param_grid = None,
        GS_scoring='accuracy', GS_verbose=2, GS_cv=5,  GS_n_splits=5,  GS_factor=2,  GS_trials=5,  GS_njobs=-1,

        GridSearchCV_binaries = False, binaries_opt_params=None,

        tol=1e-3, tunning_OA_threshold=1,
        
        plt_figsize = (12,5), plt_aspect = 'auto', plt_f1 = 0, plt_f2 = 1, plt_X_cols = None,
        plt_s = 50, plt_sv_s = 180, plt_sv_lw = 2, plt_xlabel = None, plt_ylabel = None, 
        plt_axes = False, plt_title = None, plt_fontsize = "small",
        plt_minx = None, plt_maxx = None, plt_miny = None, plt_maxy = None,
        plt_gap_x = 0.05, plt_gap_y = 0.05, plt_colors = None,
        plotter='SVC', plot_support=True,
        verbose=2, legend=True,
        OAA_borders = True, OAA_borders_style = 'solid', OAA_borders_lw = 2,
        OAA_borders_lst = [], 
        break_ties = False, random_state=10
    ) -> None:

        warnings.filterwarnings("ignore", message="No contour levels were found within the data range.")
        warnings.simplefilter(action='ignore', category=FutureWarning)

        if C is None or C <= 0.01:
            raise ValueError(f"C = None or C <=0 are not accepted. C: {C}")
            
        self.verbose = verbose
        self.random_state = random_state
        
        # CLASSIFIERS
        self.clf = None
        self.AGG_clfs = []
        self.clf_xy = None

        # PARAMETERS
        self.kernel = kernel
        self.strategy = strategy
        self.C = float(C)
        self.gamma = gamma # Only significant in kernels 'rbf', 'polynomial' and 'sigmoid'
        self.degree = degree # Only significant in kernel 'polynomial'
        self.coef0 = coef0 # Only significant in kernels 'polynomial' and 'sigmoid'        

        self.dim = None
        self.classes = []
        self.class_mean = []
        self.class_cov = []

        self.n_classes = None
        self.AGG_n_classes = None
        self.distance_method = distance_method
        self.AGG_dist_threshold = AGG_dist_threshold
        self.break_ties = break_ties
        self.AGG_default_strategy = AGG_default_strategy

        # WORKFLOW
        self.HAGG_N = HAGG_N
        self.AGG_classify_original = AGG_classify_original
        self.scale_data = scale_data
        self.interpolate_nan = interpolate_nan
        self.distance_restriction = distance_restriction

        # OPTIMIZATION
        self.GridSearchCV = GridSearchCV
        self.halving = halving
        self.GS_scoring = GS_scoring
        self.GS_verbose = GS_verbose       
        self.GS_cv = GS_cv
        self.GS_n_splits = GS_n_splits
        self.GS_factor = GS_factor
        self.GS_trials = GS_trials
        self.GS_njobs = GS_njobs
        self.best_params = []

        self.GridSearchCV_binaries = GridSearchCV_binaries
        self.binaries_opt_params = binaries_opt_params

        # STOPPING CRITERIA
        self.tol = tol
        self.tunning_OA_threshold = float(tunning_OA_threshold)

        # MONITORING
        self.time = dict()

        # LIST OF DATA AND LABELS
        self.X = None
        self.Y = None
        self.X_valid = None
        self.y_valid = None
        self.x = []
        self.y = []
        self.c1 = []
        self.c2 = []
        self.sv = []
        self.sv_y = []
        self.margins = []
        # list of merged classes [(a,b), (c,e)]
        self.superclasses = dict()
        self.superclass_tests = []
        self.superclass_tests_summary = []
        self.superclass_classifiers = []
        self.superclass_per_classifier = []
        self.superclass_best_params = []

        if param_grid is None:
            self.param_grid = self.get_param_grid()
        else:
            self.param_grid = param_grid

        self.extra_params = dict()

        # PLOTTING PARAMETERS
        self.plotter = plotter
        self.plot_support = plot_support

        self.legend = legend
        self.plt_figsize = plt_figsize
        self.plt_aspect = plt_aspect
        self.plt_X_cols = plt_X_cols
        self.plt_f1 = plt_f1
        self.plt_f2 = plt_f2
        self.plt_s = plt_s
        self.plt_sv_s = plt_sv_s
        self.plt_sv_lw = plt_sv_lw
        self.plt_xlabel = plt_xlabel
        self.plt_ylabel = plt_ylabel
        self.plt_axes = plt_axes
        self.plt_title = plt_title
        self.plt_fontsize = plt_fontsize
        self.plt_gap_x = plt_gap_x
        self.plt_gap_y = plt_gap_y
        if (plt_minx is None or plt_maxx is None or plt_miny is None or plt_maxy is None):
            self.plt_minx = None
            self.plt_maxx = None
            self.plt_miny = None
            self.plt_maxy = None
        else:
            self.plt_minx = plt_minx
            self.plt_maxx = plt_maxx
            self.plt_miny = plt_miny
            self.plt_maxy = plt_maxy
        
        # meshgrid for plotting multiclass boundaries
        self.plt_xx = None
        self.plt_yy = None
        self.plt_Z = None
        self.plt_xy = None

        # colors
        if plt_colors is None:
            plt_colors = ['blue', 'red', 'darkgreen', 'goldenrod', 'cornflowerblue', 'lightcoral', 'springgreen', 
                        'cyan', 'lightsalmon', 'lime', 'darkcyan', 'lightslategray', 'blueviolet', 'magenta',
                        'pink', 'silver', 'black']
        self.plt_colors = plt_colors

        self.OAA_borders = OAA_borders
        self.OAA_borders_style = OAA_borders_style
        self.OAA_borders_lw = OAA_borders_lw
        self.OAA_borders_lst = OAA_borders_lst

    def get_param_grid(self) -> list:
        #list_kernels = ['rbf', 'linear', 'poly']

        #list_C = np.unique(np.concatenate([np.logspace(-2, 1, 10), np.logspace(1, 3, 10)]))
        #list_C = np.array([0.01, 0.1, 1, 5, 10, 50, 100, 1000])
        list_C = np.array([0.1, 10, 25, 50, 75, 100, 150, 200, 250])

        #list_gammas = np.logspace(-9, 3, 13) #13 valores entre [1e-9, 1e3]
        #list_gammas = np.array([1e-9, 1e-4, 1e-2, 0.1, 1, 5])
        #list_gammas = np.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-2, 1e-1])
        #list_gammas = np.array([5e-4, 1e-4, 1e-2, 1])
        list_gammas = np.array([1e-2, 1, 0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2, 2.5])
        """
        scale_gamma = 1/(dim * X.var())
        if scale_gamma in list_gammas:
            list_gammas = np.delete(list_gammas, np.where(list_gammas == scale_gamma)[0][0])
        auto_gamma = 1/dim
        if auto_gamma in list_gammas:
            list_gammas = np.delete(list_gammas, np.where(list_gammas == auto_gamma)[0][0])
        list_gammas = np.insert(list_gammas, 0, scale_gamma)
        list_gammas = np.insert(list_gammas, 1, auto_gamma)
        """
        # Degree (Only significant in kernel 'polynomial')
        #list_degrees = np.arange(start=2, stop=5, step=1) #arange: intervalo aberto no final
        list_degrees = np.arange(start=2, stop=4, step=1) #arange: intervalo aberto no final

        ## Coef0 (Only significant in kernels 'polynomial' and 'sigmoid')
        #coef0 = np.array([0.5]) 
        #list_coef0 = np.arange(start=0, stop=2, step=1)
        list_coef0 = np.arange(start=0, stop=3, step=0.5)

        ## Class weight:
        """
        Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed 
        to have weight one. The “balanced” mode uses the values of y to automatically adjust weights 
        inversely proportional to class frequencies in the input data as 
        n_samples / (n_classes * np.bincount(y)).
        """
        #class_weight = np.array(['balanced', None])
        class_weight = np.array(['balanced'])

        #param_grid = dict(kernel=list_kernels, C=list_C, gamma=list_gammas, degree=list_degrees, coef0=list_coef0, class_weight=class_weight)

        dict_linear = dict(kernel=['linear'], C=list_C, class_weight=class_weight)
        dict_poly = dict(kernel=['poly'], C=list_C, degree=list_degrees, gamma=list_gammas, coef0=list_coef0, class_weight=class_weight)
        dict_rbf = dict(kernel=['rbf'], C=list_C, gamma=list_gammas, class_weight=class_weight)
        #dict_sigmoid = dict(kernel=['sigmoid'], C=list_C, gamma=list_gammas, coef0=list_coef0, class_weight=class_weight)
        
        #param_grid = [dict_rbf, dict_linear, dict_poly]
        #param_grid = [dict_rbf, dict_poly]
        param_grid = [dict_rbf]

        return param_grid


    def get_plotting_ranges(self):

        minx = min(self.X[:,self.plt_f1])
        miny = min(self.X[:,self.plt_f2])
        maxx = max(self.X[:,self.plt_f1])
        maxy = max(self.X[:,self.plt_f2])
        
        gapx = (maxx - minx) * self.plt_gap_x
        gapy = (maxy - miny) * self.plt_gap_y
        minx -= gapx
        miny -= gapy
        maxx += gapx
        maxy += gapy

        self.plt_minx = minx
        self.plt_maxx = maxx
        self.plt_miny = miny
        self.plt_maxy = maxy


    def get_data_params(self):
        
        # BASIC PARAMS
        self.class_mean = [np.array(np.mean(self.X[self.Y==c], axis=0)) for c in self.classes]
        self.class_cov = [np.cov(self.X[self.Y==c], rowvar=False) for c in self.classes]

        gamma_auto = 1/self.dim
        gamma_scale = 1/(self.dim * self.X.var())
        if self.kernel != 'linear':
            if not isinstance(self.gamma, (int, np.long, float, complex)):
                if self.gamma == 'auto':
                    # 'auto' gamma = 1 / n_features
                    self.gamma = gamma_auto
                elif self.gamma == 'scale':
                    # 'scale' gamma = 1 / (n_features * X.var())
                    self.gamma = gamma_scale
                else:
                    raise ValueError(f"Value of '{self.kernel}' not accepted for the gamma parameter")

        # PARAM GRID
        for i, g in enumerate(self.param_grid):
            if 'gamma' in g.keys():
                if gamma_scale in g['gamma']:
                    self.param_grid[i]['gamma'] = np.delete(g['gamma'], np.where(g['gamma'] == gamma_scale)[0][0])
                
                if gamma_auto in g['gamma']:
                    self.param_grid[i]['gamma'] = np.delete(g['gamma'], np.where(g['gamma'] == gamma_auto)[0][0])
                
                self.param_grid[i]['gamma'] = np.insert(self.param_grid[i]['gamma'], 0, gamma_scale)
                #self.param_grid[i]['gamma'] = np.insert(self.param_grid[i]['gamma'], 1, gamma_auto)

        # EXTRA PARAMS
        extra_params = dict()
        if self.kernel == 'rbf':
            extra_params['gamma'] = self.gamma
        elif self.kernel == 'sigmoid':
            extra_params['gamma'] = self.gamma
            extra_params['coef0'] = self.coef0
        elif self.kernel == 'poly':
            extra_params['gamma'] = self.gamma
            extra_params['degree'] = self.degree
            extra_params['coef0'] = self.coef0
        extra_params['tol'] = self.tol

        self.extra_params = extra_params

        # HAGG
        if self.strategy == 'HAGG':
            # NOTE: Number of levels must be provided for the Hierarquical Agglomerative Strategy (HAGG)
            max_levels = self.n_classes * (self.n_classes-1) / 2

            if not isinstance(self.HAGG_N, int):
                raise ValueError(f"Number of levels required for the HAGG strategy. Provide an integer value lower than or equal to {max_levels}")
            elif self.HAGG_N is None or self.HAGG_N > max_levels:
                self.HAGG_N = max_levels


    def fit(
        self, X, y, X_valid=None, y_valid=None
    ):

        start = time.time()

        self.X = X
        self.Y = y
        self.dim = X.shape[1]
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)

        self.get_data_params()

        if X_valid is not None and y_valid is not None:
            self.X_valid = X_valid
            self.y_valid = y_valid
        else:
            self.X_valid = None
            self.y_valid = None
        

        # determine plotting ranges        
        if (self.plt_minx is None or self.plt_maxx is None or
                self.plt_miny is None or self.plt_maxy is None):
            self.get_plotting_ranges()

        
        # One-Against-One (OAO) strategy
        if self.strategy == 'OAO' or self.n_classes == 2:
            self.OAO_strategy(X, y)

        elif self.strategy == 'OAA':
            self.OAA_strategy(X, y)

        elif self.strategy == 'AGG':
            self.AGG_strategy(X, y)
        
        elif self.strategy == 'PAGG':
            self.PAGG_strategy(X, y)
        
        elif self.strategy == 'HAGG':
            # Hierarquical Agglomerative Strategy
            self.HAGG_strategy(X, y)
        
        elif self.strategy == 'mPAGG':
            self.mPAGG_strategy(X, y)

        else:
            raise ValueError(
                "Strategy not found.\n"
                + "The implemented multiclass classification strategies are: OAO, OAA, AGG, PAGG and HAGG"
            )
        
        self.time['fit'] = time.time() - start

    
    #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html
    #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    def model_selection(self, clf, X, y, param_grid, halving=True):#, scoring='accuracy', verbose=2, random_state=10):#, n_splits=3, test_size=0.3, random_state=42):

        #cv = self.GS_cv #5
        #cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

        # n_jobs: number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
        if halving is True:
            #min_resources = self.GS_cv * len(np.unique(y)) * 2 #default
            min_resources = 'exhaust'
            #max_resources = X.shape[0]
            max_resources = 'auto'

            factor = self.GS_factor

            if self.X_valid is None or self.y_valid is None:
                # TODO: include the parameter 'random_state' (i.e., seed) to produce the same results in case 
                # a different hierarchical level leads to the same class setup (both classes have already
                # been merged to the same super-class)
                grid = HalvingGridSearchCV(clf, param_grid, factor=factor, cv=self.GS_cv, scoring=make_scorer(accuracy_score), 
                                            n_jobs=self.GS_njobs, return_train_score=False, resource='n_samples', 
                                            min_resources=min_resources, max_resources=max_resources, 
                                            verbose=self.GS_verbose, random_state=self.random_state
                        )
                grid.fit(X, y)
            else:
                NUM_TRIALS = self.GS_trials
                n_splits = self.GS_n_splits

                # Loop for each trial
                grids = []
                nested_scores = np.zeros(NUM_TRIALS)
                for i in range(NUM_TRIALS):
                    try:
                        # Choose cross-validation techniques for the inner and outer loops,
                        # independently of the dataset.
                        # E.g GroupKFold, LeaveOneOut, LeaveOneGroupOut, StratifiedKFold, KFold
                        inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=i)
                        #outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=i)
                        grid = HalvingGridSearchCV(clf, param_grid, factor=factor, cv=inner_cv, scoring=make_scorer(accuracy_score), 
                                            n_jobs=self.GS_njobs, return_train_score=False, resource='n_samples', 
                                            min_resources=min_resources, 
                                            max_resources=max_resources, 
                                            verbose=False
                                )
                        grid.fit(X, y)
                    except:
                        continue

                    grids.append(grid)

                    #nested_score = cross_val_score(grid, X=X_test, y=y_test, cv=outer_cv)
                    #nested_scores[i] = nested_score.mean()
                    nested_scores[i] = accuracy_score(self.y_valid, grid.predict(self.X_valid))
                    
        else:
            grid = GridSearchCV(clf, param_grid, cv=self.GS_cv, scoring=self.GS_scoring, n_jobs=self.GS_njobs)
            grid.fit(X, y)

        return grid.best_estimator_, grid.best_params_

    
    def OAO_strategy(self, X, y):
        #print(extra_params)
        if self.GridSearchCV is False:
            # NOTE: break_ties must be False when decision_function_shape is 'ovo'
            clf = SVC(kernel=self.kernel, C=self.C, decision_function_shape='ovo', break_ties=False, **self.extra_params)
        else:
            base_clf = SVC(decision_function_shape='ovo', break_ties=False)
            clf, best_params_ = self.model_selection(base_clf, X, y, param_grid=self.param_grid, halving=self.halving)
            self.best_params.append(best_params_)

        clf.fit(X, y)

        if self.clf is None:
            self.clf = clf

        if self.plotter == 'plot_margin':
            # https://scikit-learn.org/stable/modules/svm.html#svm-multi-class
            # In the case of “one-vs-one” SVC and NuSVC... The order for classes 0 to n is “0 vs 1”, “0 vs 2” , … “0 vs n”, “1 vs 2”, “1 vs 3”, “1 vs n”, . . . “n-1 vs n”.
            for i, (c1, c2) in enumerate(list(itertools.combinations(np.unique(y), 2) )):
                self.c1.append(c1)
                self.c2.append(c2)

                slice = np.logical_or(y==c1, y==c2)
                x_ = X[slice]
                temp_labels = y[slice].copy()
                
                temp_labels[y[slice]==c1] = +1
                temp_labels[y[slice]==c2] = -1

                """
                clf fornece os SVs para todas as classificações binárias. Por isso, filtramos os SVs
                considerando apenas os SVs das classes analisadas c1 e c2.
                    - clf.support_vectors_: Valores dos elementos em definidos como SVs
                    - clf.support_: índices dos elementos em X que foram definidos como SVs
                    - slice[clf.support_]: filtro que indica quais índices referem-se a elementos das classes c1 e c2 (conforme y)
                    - clf.support_vectors_[slice[clf.support_]]: VALORES dos elementos em X, referentes às classes c1 e c2, que foram definidos como SVs
                """
                sv_i = clf.support_vectors_[slice[clf.support_]]
                
                # class labels of the elements described in sv_i
                # Seleciona todos os elementos das classas c1 e c2 definidos como SVs, mesmo que em outro problema binário (não apenas em c1 vs c2)
                sv_i_labels = y[clf.support_[slice[clf.support_] == True]]
                sv_i_temp_labels = sv_i_labels.copy()
                sv_i_temp_labels[sv_i_labels==c1] = +1
                sv_i_temp_labels[sv_i_labels==c2] = -1
                sv_y_i = sv_i_temp_labels

                self.x.append(x_)
                self.y.append(temp_labels)
                self.sv.append(sv_i)
                self.sv_y.append(sv_y_i)

                #if self.dim == 2:
                if self.verbose and self.dim == 2:
                    print(f"OAO_strategy: class {c1} x class {c2}")
                    self.plot_margin(clf, i, x_, temp_labels, c1, c2=c2, sv=sv_i, plot_support=self.plot_support, 
                                        figsize=self.plt_figsize, aspect=self.plt_aspect, xlabel=self.plt_xlabel, ylabel=self.plt_ylabel, 
                                        axes=self.plt_axes, title=self.plt_title, fontsize=self.plt_fontsize)
                    #self.plot_svc_decision_function(clf, i, x_[temp_labels == 1], x_[temp_labels == -1], c1, figsize=figsize, aspect=aspect)
        else:
            if self.verbose and self.dim == 2:
                self.plot_svc_subproblems(clf, X, y, plot_support=self.plot_support, colors=self.plt_colors,
                                            xlabel=self.plt_xlabel, ylabel=self.plt_ylabel, axes=self.plt_axes, 
                                            title=self.plt_title, fontsize=self.plt_fontsize)

    def OAA_strategy(self, X, y):
        #print(extra_params)
        if self.GridSearchCV is False:
            # OneVsRestClassifier: use 'n_jobs=-1' for parallel processing
            clf = OneVsRestClassifier(SVC(kernel=self.kernel, C=self.C, break_ties=self.break_ties, **self.extra_params))
        else:
            # OneVsRestClassifier: use 'n_jobs=-1' for parallel processing
            base_clf = OneVsRestClassifier(SVC(break_ties=self.break_ties))
            param_grid_OvR = []
            for i, d_orig in enumerate(self.param_grid):
                d_new = dict()
                for k, v in d_orig.items():
                    d_new[f"estimator__{k}"] = v
                param_grid_OvR.append(d_new)
            clf, best_params_ = self.model_selection(base_clf, X, y, param_grid=param_grid_OvR, halving=self.halving)
            self.best_params.append(best_params_)
            
        clf.fit(X, y)
        clf.kernel=self.kernel #OneVsRestClassifier does not have the kernel information. It is informed inside each 'estimator_'
        
        if self.clf is None:
            self.clf = clf

        if self.plotter == 'plot_margin':
            temp_labels_0_1 = clf.label_binarizer_.transform(y).toarray().T

            for i, (clf_, temp_labels) in enumerate(zip(clf.estimators_, temp_labels_0_1)):
                c1 = clf.classes_[i]
                self.c1.append(c1)
                self.c2.append('rest')

                temp_labels[temp_labels==0] = -1

                self.x.append(X)
                self.y.append(temp_labels)
                sv = clf_.support_vectors_
                sv_y = temp_labels[clf_.support_]
                self.sv.append(sv)
                self.sv_y.append(sv_y)

                #if self.dim == 2:
                if self.verbose and self.dim == 2:
                    print(f"OAA_strategy: class {c1} x rest")
                    self.plot_margin(clf, i, X, temp_labels, c1, sv=sv, plot_support=self.plot_support, 
                                        figsize=self.plt_figsize, aspect=self.plt_aspect, xlabel=self.plt_xlabel, ylabel=self.plt_ylabel, 
                                        axes=self.plt_axes, title=self.plt_title, fontsize=self.plt_fontsize)
                    #self.plot_svc_decision_function(clf_, i, X[temp_labels == 1], X[temp_labels == -1], c1, figsize=figsize, aspect=aspect)
        else:
            if self.verbose and self.dim == 2:
                self.plot_svc_subproblems(clf, X, y, plot_support=self.plot_support, colors=self.plt_colors,
                                            xlabel=self.plt_xlabel, ylabel=self.plt_ylabel, axes=self.plt_axes, 
                                            title=self.plt_title, fontsize=self.plt_fontsize)

    def PAGG_strategy(self, X, y):
    
        ## --- CLASS SEPARABILITY ANALYSIS ---
        df_class_separability = self.class_separability_analysis(method=self.distance_method)

        self.AGG_class_separability_original = df_class_separability
        
        df_class_separability, _ = self.filter_separability_dataframe(df_class_separability, two_class_clusters=True, distance_restriction=self.distance_restriction)

        OAA_svm_model = AGG_SVC(strategy='OAA', kernel=self.kernel, C=self.C, gamma=self.gamma, degree=self.degree, coef0=self.coef0, scale_data=False, 
                                interpolate_nan=self.interpolate_nan, distance_method=self.distance_method, 
                                GridSearchCV = self.GridSearchCV, halving = self.halving, param_grid = self.param_grid,
                                GS_scoring=self.GS_scoring, GS_verbose=self.GS_verbose, GS_cv=self.GS_cv,  GS_n_splits=self.GS_n_splits,  
                                GS_factor=self.GS_factor,  GS_trials=self.GS_trials,  GS_njobs=self.GS_njobs,
                                tol=self.tol, tunning_OA_threshold=self.tunning_OA_threshold,
                                plt_figsize = self.plt_figsize, plt_aspect = self.plt_aspect, plt_f1 = self.plt_f1, plt_f2 = self.plt_f2, plt_X_cols = self.plt_X_cols,
                                plt_s = self.plt_s, plt_sv_s = self.plt_sv_s, plt_sv_lw = self.plt_sv_lw, plt_xlabel = self.plt_xlabel, plt_ylabel = self.plt_ylabel, 
                                plt_axes = self.plt_axes, plt_title = self.plt_title, plt_fontsize = self.plt_fontsize,
                                plt_minx = self.plt_minx, plt_maxx = self.plt_maxx, plt_miny = self.plt_miny, plt_maxy = self.plt_maxy,
                                plt_gap_x = self.plt_gap_x, plt_gap_y = self.plt_gap_y, plt_colors = self.plt_colors,
                                plotter=self.plotter, plot_support=self.plot_support, legend=self.legend,
                                verbose=self.verbose, break_ties = self.break_ties, random_state=self.random_state)
        
        if self.AGG_classify_original is True:
            OAA_svm_model_ = deepcopy(OAA_svm_model)

            OAA_svm_model_.fit(X, y)
            self.AGG_clfs.append(OAA_svm_model_)
            # concatenates lists of margins
            self.margins = self.margins + OAA_svm_model_.margins
            if self.verbose is True:
                OAA_svm_model_.show_dataset(multiclass_boundary=True, aspect='auto')

        y_ = y.copy()
        
        # TODO: merge pairs of classes
        classes_remap = {}
        num_superclasses = 0

        for i, row in df_class_separability.iterrows():
            # Ignore the pair of classes in case any of them have already been merged with another class
            if row['c1'] in classes_remap.keys() or row['c2'] in classes_remap.keys():
                df_class_separability = df_class_separability.drop([i])
                continue
            
            num_superclasses +=1
            # We use the label of c1 to the superclass
            classes_remap[row['c1']] = row['c1'] #f"S{num_SuperClasses}"
            classes_remap[row['c2']] = row['c1'] #f"S{num_SuperClasses}"

            #print("\n")
            #print(row['c1'], row['c2'])

            classes_temp = {}
            for k in self.classes:
                if k in classes_remap.keys():
                    classes_temp[k] = classes_remap[k]
                else:
                    classes_temp[k] = k

            def group_by_superclass(d):
                result = {}
                for k, v in d.items():
                    result.setdefault(v, []).append(k)

                ordered_result = {}
                for k, v in result.items():
                    # NOTE: estava min(v) ao invés de d[k]. 
                    # Corrigido por dar erro em um caso: {2.0: [2., 1., 3.]}. 
                    # A superclasse nesse caso foi definida como 2 (não como 1, que é o mínimo)
                    ordered_result[d[k]] = np.sort(v)
                
                return ordered_result

            superclasses = group_by_superclass(classes_temp)
            #print(superclasses)

            for superclass_label, original_labels in superclasses.items():
                SC_indices = np.argwhere(np.isin(y, original_labels)).ravel()
                y_[SC_indices] = superclass_label
            
            AGG_unique_labels = np.unique(y_)
            self.AGG_n_classes = len(AGG_unique_labels)

            # TODO: OAA classification
            #self.OAA_strategy(X, y_, gridSearch=gridSearch, extra_params=extra_params, figsize=figsize, aspect=aspect)

            OAA_svm_model_ = deepcopy(OAA_svm_model)

            #OAA_svm_model_.fit(df, label_column=label_column, f1=0, f2=1)
            OAA_svm_model_.fit(X, y_)

            self.AGG_clfs.append(OAA_svm_model_)
            # concatenates lists of margins
            self.margins = self.margins + OAA_svm_model_.margins

            if self.verbose is True:
                OAA_svm_model_.show_dataset(multiclass_boundary=True, aspect='auto')

            #self.x = []
            #self.y = []
            #self.sv = []
            #self.sv_y = []

            n_candidates = 1
            for params in self.param_grid:
                n_candidates *= len(params.values())
            print(f"n_candidates: {n_candidates}")

            c1 = row['c1']
            c2 = row['c2']
            superclass_label = row['c1']
            
            self.superclasses[superclass_label] = np.array([row['c1'], row['c2']])

            # TODO: identify the best parameters to classify the pair of classes merged in the current iteration
            print(f"Fitting parameters for superclass ({c1}, {c2})")

            c1_c2_indices = np.argwhere(np.isin(y, [c1, c2])).ravel()
            X_binary = X[c1_c2_indices]
            y_binary = y[c1_c2_indices]

            #--
            pg = self.param_grid.copy()
            gamma_scale = 1/(self.dim * self.X.var())
            gamma_scale_ = 1/(self.dim * X_binary.var())
            for i, g in enumerate(pg):
                if 'gamma' in g.keys():
                    if gamma_scale in g['gamma']:
                        pg[i]['gamma'] = np.delete(g['gamma'], np.where(g['gamma'] == gamma_scale)[0][0])
                    
                    if gamma_scale_ in g['gamma']:
                        pg[i]['gamma'] = np.delete(g['gamma'], np.where(g['gamma'] == gamma_scale_)[0][0])
                    
                    pg[i]['gamma'] = np.insert(pg[i]['gamma'], 0, gamma_scale_)
                    pg[i]['gamma'] = np.insert(pg[i]['gamma'], 1, gamma_scale)
            #--

            base_clf = SVC(decision_function_shape='ovo', break_ties=False)
            clf, best_params_ = self.model_selection(base_clf, X_binary, y_binary, param_grid=pg, halving=True)

            clf.fit(X_binary, y_binary)

            self.superclass_classifiers.append(clf)
            self.superclass_per_classifier.append(superclass_label)
            self.superclass_best_params.append(best_params_)

            # PLOT
            if self.plotter == 'plot_margin':
                # https://scikit-learn.org/stable/modules/svm.html#svm-multi-class
                # In the case of “one-vs-one” SVC and NuSVC... The order for classes 0 to n is “0 vs 1”, “0 vs 2” , … “0 vs n”, “1 vs 2”, “1 vs 3”, “1 vs n”, . . . “n-1 vs n”.
                
                #self.c1.append(c1)
                #self.c2.append(c2)

                slice = np.logical_or(y_binary==c1, y_binary==c2)
                x_ = X_binary[slice]
                temp_labels = y_binary[slice].copy()
                #temp_labels = y_binary.copy()
                
                temp_labels[y_binary[slice]==c1] = +1
                temp_labels[y_binary[slice]==c2] = -1

                """
                clf fornece os SVs para todas as classificações binárias. Por isso, filtramos os SVs
                considerando apenas os SVs das classes analisadas c1 e c2.
                    - clf.support_vectors_: Valores dos elementos em definidos como SVs
                    - clf.support_: índices dos elementos em X que foram definidos como SVs
                    - slice[clf.support_]: filtro que indica quais índices referem-se a elementos das classes c1 e c2 (conforme y)
                    - clf.support_vectors_[slice[clf.support_]]: VALORES dos elementos em X, referentes às classes c1 e c2, que foram definidos como SVs
                """
                sv_i = clf.support_vectors_[slice[clf.support_]]
                
                # class labels of the elements described in sv_i
                # Seleciona todos os elementos das classas c1 e c2 definidos como SVs, mesmo que em outro problema binário (não apenas em c1 vs c2)
                sv_i_labels = y_binary[clf.support_[slice[clf.support_] == True]]
                sv_i_temp_labels = sv_i_labels.copy()
                sv_i_temp_labels[sv_i_labels==c1] = +1
                sv_i_temp_labels[sv_i_labels==c2] = -1
                sv_y_i = sv_i_temp_labels
                
                self.x.append(x_)
                self.y.append(temp_labels)
                self.sv.append(sv_i)
                self.sv_y.append(sv_y_i)

                #if self.dim >= 2:
                if self.verbose and self.dim == 2:
                    print(f"OAO_strategy: class {c1} x class {c2}")
                    self.plot_margin(clf, 1, x_, temp_labels, c1, c2=c2, sv=sv_i, plot_support=self.plot_support, 
                                        figsize=self.plt_figsize, aspect=self.plt_aspect, xlabel=self.plt_xlabel, ylabel=self.plt_ylabel, 
                                        axes=self.plt_axes, title=self.plt_title, fontsize=self.plt_fontsize)
                    #self.plot_margin(clf, 1, x_[temp_labels == 1], x_[temp_labels == -1], c1, c2=c2, figsize=figsize, aspect=aspect)
                    
            else:
                if self.verbose and self.dim == 2:
                    self.plot_svc_subproblems(clf, X_binary, y_binary, plot_support=self.plot_support, colors=self.plt_colors,
                                                xlabel=self.plt_xlabel, ylabel=self.plt_ylabel, axes=self.plt_axes, 
                                                title=self.plt_title, fontsize=self.plt_fontsize)
        
        self.AGG_class_separability = df_class_separability
        

    def mPAGG_strategy(self, X, y):
        n_clfs = 0
        ## --- CLASS SEPARABILITY ANALYSIS ---
        df_class_separability = self.class_separability_analysis(method=self.distance_method)

        self.AGG_class_separability_original = df_class_separability
        
        df_class_separability, classes_remap = self.filter_separability_dataframe(df_class_separability, two_class_clusters=False, distance_restriction=self.distance_restriction)
        self.classes_remap = classes_remap

        OAA_svm_model = AGG_SVC(strategy=self.AGG_default_strategy, kernel=self.kernel, C=self.C, gamma=self.gamma, degree=self.degree, coef0=self.coef0, scale_data=False, 
                                interpolate_nan=self.interpolate_nan, distance_method=self.distance_method, AGG_dist_threshold=self.AGG_dist_threshold,
                                GridSearchCV = self.GridSearchCV, halving = self.halving, param_grid = self.param_grid,
                                GS_scoring=self.GS_scoring, GS_verbose=self.GS_verbose, GS_cv=self.GS_cv,  GS_n_splits=self.GS_n_splits,  
                                GS_factor=self.GS_factor,  GS_trials=self.GS_trials,  GS_njobs=self.GS_njobs,
                                tol=self.tol, tunning_OA_threshold=self.tunning_OA_threshold,
                                plt_figsize = self.plt_figsize, plt_aspect = self.plt_aspect, plt_f1 = self.plt_f1, plt_f2 = self.plt_f2, plt_X_cols = self.plt_X_cols,
                                plt_s = self.plt_s, plt_sv_s = self.plt_sv_s, plt_sv_lw = self.plt_sv_lw, plt_xlabel = self.plt_xlabel, plt_ylabel = self.plt_ylabel, 
                                plt_axes = self.plt_axes, plt_title = self.plt_title, plt_fontsize = self.plt_fontsize,
                                plt_minx = self.plt_minx, plt_maxx = self.plt_maxx, plt_miny = self.plt_miny, plt_maxy = self.plt_maxy,
                                plt_gap_x = self.plt_gap_x, plt_gap_y = self.plt_gap_y, plt_colors = self.plt_colors,
                                plotter=self.plotter, plot_support=self.plot_support, legend=self.legend,
                                verbose=self.verbose, break_ties = self.break_ties, random_state=self.random_state)
        
        #if self.AGG_classify_original is True:
        #    n_clfs = n_clfs + 1
        #    print(f"clf{n_clfs} - {self.AGG_default_strategy}")
        #    OAA_svm_model_ = deepcopy(OAA_svm_model)
        #
        #    OAA_svm_model_.fit(X, y)
        #    self.clf = OAA_svm_model_
        #    #self.AGG_clfs.append(OAA_svm_model_)
        #    # concatenates lists of margins
        #    self.margins = self.margins + OAA_svm_model_.margins
        #    if self.verbose is True:
        #        OAA_svm_model_.show_dataset(multiclass_boundary=True, aspect=self.plt_aspect)


        ## [mPAGG] Passo 1: DEFINIR SUPERCLASSES E ORDEM CORRETA DE ANÁLISE
        def group_by_superclass(d, df_class_separability):
            result = {}
            for k, v in d.items():
                result.setdefault(v, []).append(k)

            ordered_result = {}
            for k, v in result.items():
                ordered_result[d[k]] = np.sort(v)
            
            # PRESERVING THE SORTING FROM THE SEPARABILITY ANALYSIS
            # identifica a ordem em que as classes aparecem no dataframe de separabilidade
            v = df_class_separability[['c1', 'c2']].values.ravel()
            # encontra as superclasses correspondentes
            scs = [d[x] for x in v]
            # lista as classes únicas, preservando a ordem
            indexes = np.unique(scs, return_index=True)[1]
            scs_sorted_by_dist = [scs[index] for index in sorted(indexes)]

            superclasses = {v: ordered_result[v] for v in scs_sorted_by_dist}

            return superclasses

        superclasses = group_by_superclass(classes_remap, df_class_separability)
        print("superclasses ", superclasses)

        y_ = y.copy()
        #num_superclasses = len(superclasses)

        """
        ## NOTE: PARA INSERIR CLASSIFICAÇÃO OAA ENTRE CADA MERGE (NÃO APRESENTOU BONS RESULTADOS)
        for sc, sc_classes in superclasses.items():
            print(sc, sc_classes)
            self.superclasses[sc] = np.array(sc_classes)

            SC_indices = np.argwhere(np.isin(y, sc_classes)).ravel()
            y_[SC_indices] = sc

            AGG_unique_labels = np.unique(y_)
            #self.AGG_n_classes = len(AGG_unique_labels) # REVIEW

            if len(AGG_unique_labels) > 1:
                n_clfs = n_clfs + 1
                print(f"clf{n_clfs} - OAA (merged)")

                OAA_svm_model_ = deepcopy(OAA_svm_model)
                OAA_svm_model_.fit(X, y_)
                self.AGG_clfs.append(OAA_svm_model_) #TODO
                self.margins = self.margins + OAA_svm_model_.margins #TODO
                if self.verbose is True: #TODO
                    OAA_svm_model_.show_dataset(multiclass_boundary=True, aspect='auto')

            self.x = [] #TODO
            self.y = [] #TODO
            self.sv = [] #TODO
            self.sv_y = [] #TODO

            for i, row in df_class_separability.loc[
                (df_class_separability['c1'].isin(sc_classes)) | (df_class_separability['c2'].isin(sc_classes))
            ].iterrows():
                n_clfs = n_clfs + 1
                c1, c2 = row['c1'], row['c2']
                print(f"clf{n_clfs} - binary")
                print(f"Fitting parameters for superclass ({c1}, {c2})")
        """

        #OAO_svm_model = AGG_SVC(strategy='OAO', kernel=self.kernel, C=self.C, gamma=self.gamma, degree=self.degree, coef0=self.coef0, scale_data=False, 
        #                        interpolate_nan=self.interpolate_nan, distance_method=self.distance_method, AGG_dist_threshold=self.AGG_dist_threshold,
        #                        GridSearchCV = self.GridSearchCV, halving = self.halving, param_grid = self.param_grid,
        #                        GS_scoring=self.GS_scoring, GS_verbose=self.GS_verbose, GS_cv=self.GS_cv,  GS_n_splits=self.GS_n_splits,  
        #                        GS_factor=self.GS_factor,  GS_trials=self.GS_trials,  GS_njobs=self.GS_njobs,
        #                        tol=self.tol, tunning_OA_threshold=self.tunning_OA_threshold,
        #                        plt_figsize = self.plt_figsize, plt_aspect = self.plt_aspect, plt_f1 = self.plt_f1, plt_f2 = self.plt_f2, plt_X_cols = self.plt_X_cols,
        #                        plt_s = self.plt_s, plt_sv_s = self.plt_sv_s, plt_sv_lw = self.plt_sv_lw, plt_xlabel = self.plt_xlabel, plt_ylabel = self.plt_ylabel, 
        #                        plt_axes = self.plt_axes, plt_title = self.plt_title, plt_fontsize = self.plt_fontsize,
        #                        plt_minx = self.plt_minx, plt_maxx = self.plt_maxx, plt_miny = self.plt_miny, plt_maxy = self.plt_maxy,
        #                        plt_gap_x = self.plt_gap_x, plt_gap_y = self.plt_gap_y, plt_colors = self.plt_colors,
        #                        plotter=self.plotter, plot_support=self.plot_support,
        #                        verbose=self.verbose, break_ties = self.break_ties, random_state=self.random_state)

        self.superclasses = superclasses
        
        # Agrupando as superclasses primeiramente
        for sc, sc_classes in superclasses.items():
            if self.verbose is not False:
                print(sc, sc_classes)
            #self.superclasses[sc] = np.array(sc_classes)

            SC_indices = np.argwhere(np.isin(y, sc_classes)).ravel()
            y_[SC_indices] = sc

        AGG_unique_labels = np.unique(y_)
        #self.AGG_n_classes = len(AGG_unique_labels) # REVIEW

        if len(AGG_unique_labels) > 1:
            n_clfs = n_clfs + 1
            #print(f"clf{n_clfs} - {self.AGG_default_strategy} (merged)")

            OAA_svm_model_ = deepcopy(OAA_svm_model)
            OAA_svm_model_.fit(X, y_)
            self.clf = OAA_svm_model_
            #self.AGG_clfs.append(OAA_svm_model_) #TODO
            self.margins = self.margins + OAA_svm_model_.margins #TODO
            if self.verbose is True: #TODO
                OAA_svm_model_.show_dataset(multiclass_boundary=True, aspect='auto')

        self.x = [] #TODO
        self.y = [] #TODO
        self.sv = [] #TODO
        self.sv_y = [] #TODO

        # for i, row in df_class_separability.loc[
        #     (df_class_separability['c1'].isin(sc_classes)) | (df_class_separability['c2'].isin(sc_classes))
        # ].iterrows():
        #for i, row in df_class_separability.iterrows():
            #c1, c2 = row['c1'], row['c2']
        for sc, sc_classes in superclasses.items():
            print(f"Superclass: {sc} -> {sc_classes}")
            for i, (c1, c2) in enumerate(list(itertools.combinations(sc_classes, 2) )):
                n_clfs = n_clfs + 1
                #print(f"clf{n_clfs} - binary")
                print(f"Fitting parameters for superclass ({c1}, {c2})")

                #self.superclasses[sc] = np.array([c1, c2])

                c1_c2_indices = np.argwhere(np.isin(y, [c1, c2])).ravel()
                X_binary = X[c1_c2_indices]
                y_binary = y[c1_c2_indices]

                #--
                pg = self.param_grid.copy()
                gamma_scale = 1/(self.dim * self.X.var())
                gamma_scale_ = 1/(self.dim * X_binary.var())
                for i, g in enumerate(pg):
                    if 'gamma' in g.keys():
                        if gamma_scale in g['gamma']:
                            pg[i]['gamma'] = np.delete(g['gamma'], np.where(g['gamma'] == gamma_scale)[0][0])
                        
                        if gamma_scale_ in g['gamma']:
                            pg[i]['gamma'] = np.delete(g['gamma'], np.where(g['gamma'] == gamma_scale_)[0][0])
                        
                        pg[i]['gamma'] = np.insert(pg[i]['gamma'], 0, gamma_scale_)
                        pg[i]['gamma'] = np.insert(pg[i]['gamma'], 1, gamma_scale)
                #--

                if self.GridSearchCV_binaries is True:
                    base_clf = SVC(decision_function_shape='ovo', break_ties=False)
                    clf, best_params_ = self.model_selection(base_clf, X_binary, y_binary, param_grid=pg, halving=self.halving)
                
                elif isinstance(self.binaries_opt_params, dict) and len(self.binaries_opt_params) > 0:
                    try:
                        opt_params = self.binaries_opt_params.get(c1).get(c2)
                        if opt_params is None:
                            raise ValueError("Value not found")
                    except Exception as e:
                        raise ValueError(f"Falha ao extrair parâmetros informados para classificação otimizada ({c1}x{c2}): {e}")

                    clf = SVC(decision_function_shape='ovo', break_ties=False, **opt_params)
                    best_params_ = opt_params
                
                else:
                    clf = SVC(kernel=self.kernel, C=self.C, decision_function_shape='ovo', break_ties=False, **self.extra_params)
                    best_params_ = {}
                
                if self.verbose is not False:
                    print(clf)

                clf.fit(X_binary, y_binary)
                
                self.superclass_classifiers.append(clf)
                self.superclass_per_classifier.append(sc)
                self.superclass_best_params.append(best_params_)

                    # PLOT
                if self.plotter == 'plot_margin':
                    # https://scikit-learn.org/stable/modules/svm.html#svm-multi-class
                    # In the case of “one-vs-one” SVC and NuSVC... The order for classes 0 to n is “0 vs 1”, “0 vs 2” , … “0 vs n”, “1 vs 2”, “1 vs 3”, “1 vs n”, . . . “n-1 vs n”.
                    
                    #self.c1.append(c1)
                    #self.c2.append(c2)

                    slice = np.logical_or(y_binary==c1, y_binary==c2)
                    x_ = X_binary[slice]
                    temp_labels = y_binary[slice].copy()
                    #temp_labels = y_binary.copy()
                    
                    temp_labels[y_binary[slice]==c1] = +1
                    temp_labels[y_binary[slice]==c2] = -1

                    """
                    clf fornece os SVs para todas as classificações binárias. Por isso, filtramos os SVs
                    considerando apenas os SVs das classes analisadas c1 e c2.
                        - clf.support_vectors_: Valores dos elementos em definidos como SVs
                        - clf.support_: índices dos elementos em X que foram definidos como SVs
                        - slice[clf.support_]: filtro que indica quais índices referem-se a elementos das classes c1 e c2 (conforme y)
                        - clf.support_vectors_[slice[clf.support_]]: VALORES dos elementos em X, referentes às classes c1 e c2, que foram definidos como SVs
                    """
                    sv_i = clf.support_vectors_[slice[clf.support_]]
                    
                    # class labels of the elements described in sv_i
                    # Seleciona todos os elementos das classas c1 e c2 definidos como SVs, mesmo que em outro problema binário (não apenas em c1 vs c2)
                    sv_i_labels = y_binary[clf.support_[slice[clf.support_] == True]]
                    sv_i_temp_labels = sv_i_labels.copy()
                    sv_i_temp_labels[sv_i_labels==c1] = +1
                    sv_i_temp_labels[sv_i_labels==c2] = -1
                    sv_y_i = sv_i_temp_labels
                    
                    self.x.append(x_)
                    self.y.append(temp_labels)
                    self.sv.append(sv_i)
                    self.sv_y.append(sv_y_i)

                    #if self.dim >= 2:
                    if self.verbose and self.dim == 2:
                        print(f"Binary classification: class {c1} x class {c2}")
                        self.plot_margin(clf, 1, x_, temp_labels, c1, c2=c2, sv=sv_i, plot_support=self.plot_support, 
                                            figsize=self.plt_figsize, aspect=self.plt_aspect, xlabel=self.plt_xlabel, ylabel=self.plt_ylabel, 
                                            axes=self.plt_axes, title=self.plt_title, fontsize=self.plt_fontsize)
                        #self.plot_margin(clf, 1, x_[temp_labels == 1], x_[temp_labels == -1], c1, c2=c2, figsize=figsize, aspect=aspect)
                        
                else:
                    if self.verbose and self.dim == 2:
                        self.plot_svc_subproblems(clf, X_binary, y_binary, plot_support=self.plot_support, colors=self.plt_colors,
                                                    xlabel=self.plt_xlabel, ylabel=self.plt_ylabel, axes=self.plt_axes, 
                                                    title=self.plt_title, fontsize=self.plt_fontsize)
        self.AGG_class_separability = df_class_separability


    def filter_separability_dataframe(self, df_class_separability, two_class_clusters=True, distance_restriction=False):
        
        classes_remap = {}
        num_superclasses = 0
        
        if self.AGG_dist_threshold is not None:
            df_class_separability = df_class_separability.loc[
                df_class_separability['distance'] <= self.AGG_dist_threshold
            ]

        df_class_separability_filtered = df_class_separability.copy()

        if two_class_clusters is True:

            for i, row in df_class_separability.iterrows():
                
                if distance_restriction is True:
                    n_ignored_mergings = df_class_separability.loc[
                        ((df_class_separability['c1'] == row['c1']) | (df_class_separability['c1'] == row['c2']) | 
                            (df_class_separability['c2'] == row['c1']) | (df_class_separability['c2'] == row['c2'])) &
                        (df_class_separability['distance'] < row['distance'])
                    ].shape[0]
                else:
                    n_ignored_mergings = 0

                # Ignore the pair of classes in case any of them have already been merged with another class 
                # or the distance between the classes is larger than the distance of one of these classes to another class
                if row['c1'] in classes_remap.keys() or row['c2'] in classes_remap.keys() or (n_ignored_mergings > 0):
                    df_class_separability_filtered = df_class_separability_filtered.drop([i])
                    continue
                
                num_superclasses +=1
                # We use the label of c1 to the superclass
                classes_remap[row['c1']] = row['c1'] #f"S{num_SuperClasses}"
                classes_remap[row['c2']] = row['c1'] #f"S{num_SuperClasses}"

        else:

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
                    if classes_remap[row['c1']] == classes_remap[row['c2']]:
                        df_class_separability_filtered = df_class_separability_filtered.drop([i])
                        continue
                    else:
                        c = [classes_remap[row['c1']], classes_remap[row['c2']]]
                        c_c1 = min(c)
                        c_c2 = max(c)
                        for k, v in classes_remap.items():
                            if v == c_c2:
                                classes_remap[k] = c_c1

                elif row['c1'] in classes_remap.keys():
                    # Only c1 is already merged. Then, add c2 to the same superclass
                    classes_remap[row['c2']] = classes_remap[row['c1']]

                elif row['c2'] in classes_remap.keys():
                    # Only c2 is already merged. Then, add c1 to the same superclass
                    classes_remap[row['c1']] = classes_remap[row['c2']]

        return df_class_separability_filtered, classes_remap


    def plot_margin(self, clf_, i, X, y, c1, c2=None, sv=None, plot_support=True, figsize=(12,5), aspect='auto',
                    xlabel=None, ylabel=None, axes=False, title=None, fontsize="small"):
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
            #return (-w[f1] * x - b + c) / w[f2]
            a = -w[f1] / w[f2]
            return a * x - b

        if self.verbose:
            print(f" plotting margin...")

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
            ax.set_xlim(self.plt_minx, self.plt_maxx)
            ax.set_ylim(self.plt_miny, self.plt_maxy)

            # Axis limits.
            #x1_min, x1_max = X1.min(), X1.max()
            #x2_min, x2_max = X2.min(), X2.max()
            #ax.set(xlim=(x1_min, x1_max), ylim=(x2_min, x2_max))

            for c, a in zip([c1, c2], [1, -1]):
                #t = np.argwhere(self.y[i] == a)
                t = np.argwhere(y == a)
                t = t[:,0]
                if c is None:
                    label = 'Rest'
                    c = -1
                else:
                    label = f'Class {c}'

                clr_id = np.where(np.unique(y)==a)[0][0]
                ax.scatter(X[t, self.plt_f1], X[t, self.plt_f2], s=self.plt_s, color=self.plt_colors[clr_id], label=label);
                #ax.scatter(self.x[i][t,self.plt_f1], self.x[i][t,self.plt_f2], s=50, color=self.plt_colors[int(c)], label=label)#, edgecolor='k', linewidth=1.5)

                ## Plotting the SVs with the same color of their class
                #tsv = np.argwhere(self.sv_y[i] == a)
                #tsv = tsv[:,0]
                #ax.scatter(self.sv[i][tsv, self.plt_f1], self.sv[i][tsv, self.plt_f2], s=800, linewidth=1.8, facecolors="none", edgecolor=self.plt_colors[int(c)], label="Support vectors")
            
            # Free support vectors
            #plt.scatter(self.free_sv[i][:, self.plt_f1], self.free_sv[i][:, self.plt_f2], s=90, linewidth=1.8, facecolors="none", edgecolor="dimgray")#, label="Free SV")

            # The points designating the support vectors.
            if plot_support is True and sv is not None:
                #plt.scatter(self.sv[i][:, self.plt_f1], self.sv[i][:, self.plt_f2], s=90, linewidth=2, facecolors="none", edgecolor="k", label="Support vectors")
                plt.scatter(sv[:, self.plt_f1], sv[:, self.plt_f2], s=self.plt_sv_s, linewidth=self.plt_sv_lw, facecolors="none", edgecolor="k", label="Support vectors")
        
        # REVIEW: Funciona para OAA, mas não para OAO. No OAO os hiperplanos são plotados errados. No OAA falta ajusta as margens (pontilhados)
        if clf_.kernel  != 'linear':
            # Non-linear margin line needs to be generated. Will use a contour plot.
            
            if self.plt_xx is None or self.plt_yy is None:
                #_X1, _X2 = np.meshgrid(np.linspace(self.plt_minx, self.plt_maxx, 500), np.linspace(self.plt_minx, self.plt_maxx, 500))
                xnum, ynum = plt.gcf().dpi * plt.gcf().get_size_inches()
                xnum, ynum = math.floor(xnum), math.ceil(ynum)
                _X1, _X2 = np.meshgrid(np.linspace(self.plt_minx, self.plt_maxx, num=xnum),
                                        np.linspace(self.plt_miny, self.plt_maxy, num=ynum))
                self.plt_xx = _X1
                self.plt_yy = _X2
            else:
                _X1 = self.plt_xx
                _X2 = self.plt_yy

            #xy = np.array([[x1, x2] for x1, x2 in zip(np.ravel(_X1), np.ravel(_X2))])
            xy = np.vstack([_X1.ravel(), _X2.ravel()]).T

            #n_classes = len(clf.classes_)
            #if clf_.decision_function_shape == 'ovo':
            #    n_estimators = int(n_classes * (n_classes - 1) / 2)
            #else:
            #    n_estimators = n_classes

            #Z = self.project_features(xy, i).reshape(SX1.shape)
            if isinstance(clf_, OneVsRestClassifier):
                #Z = clf_.estimators_[i].predict(X).reshape(_X1.shape)
                Z = clf_.estimators_[i].decision_function(xy).reshape(_X1.shape)
            else:
                #Z = clf_.predict(X).reshape(_X1.shape)
                #Z = clf_.classes_[np.argmax(clf.decision_function(xy), axis=1)].reshape(XX.shape)
                if len(clf_.classes_) > 2:
                    Z = clf_.decision_function(xy)[:, i].reshape(_X1.shape)
                else:
                    Z = clf_.decision_function(xy).reshape(_X1.shape)
                

            #Z = clf_.decision_function(np.c_[np.ravel(_X1), np.ravel(_X2)]).reshape(_X1.shape)
            #Z = clf_.predict(np.c_[np.ravel(_X1), np.ravel(_X2)]).reshape(_X1.shape)
            
            if self.verbose:
                plt.contour(_X1, _X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
                # Legend
                if self.legend is True:
                    fig.legend(loc='lower center', fancybox=True, shadow=False, ncol=3,
                        title_fontsize=9, framealpha=0, labelspacing=1, edgecolor='k')#, title="Legend")

                if axes is False:
                    ax.set_xticklabels([])
                    ax.set_xticks([])
                    ax.set_yticklabels([])
                    ax.set_yticks([])
                    #ax.get_xaxis().set_visible(False)
                    #ax.get_yaxis().set_visible(False)
                    plt.gcf().set_facecolor('white')

                if xlabel is not None:
                    ax.set_xlabel(xlabel)
                else:
                    ax.set_xlabel('$x_1$')
                if ylabel is not None:
                    ax.set_ylabel(ylabel)
                else:
                    ax.set_ylabel('$x_2$')
                
                if title is not None:
                    plt.title(title, fontsize=fontsize)

                plt.show();

            self.margins.append((_X1, _X2, Z))

            # Margens erradas. Precisam ser recalculadas, ao invés de apenas utilizar Z +- 1
            #plt.contour(_X1, _X2, Z + 1, [0.0], colors='grey', linestyles='--', linewidths=1, origin='lower')
            #plt.contour(_X1, _X2, Z - 1, [0.0], colors='grey', linestyles='--', linewidths=1, origin='lower')
        else:
            ## Linear margin line needs to be generated.
            ## This section can be done by the above code and use plt.contour. But wanting to generate the linear lines here for demonstration.
            ## Decision Boundary:  w.x + b = 0

            # Separating hyperplane
            w = clf_.coef_[i]
            a = -w[self.plt_f1] / w[self.plt_f2]
            xx = np.linspace(self.plt_minx, self.plt_maxx)
            yy = a * xx - (clf_.intercept_[i]) / w[self.plt_f2]
            
            self.margins.append((xx, yy))
            
            if self.verbose:
                plt.plot(xx, yy, "k");

                # plot the parallels to the separating hyperplane that pass through the
                # support vectors (margin away from hyperplane in direction
                # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
                # 2-d.
                margin = 1 / np.sqrt(np.sum(clf_.coef_[i]**2))
                yy_down = yy - np.sqrt(1 + a**2) * margin
                yy_up = yy + np.sqrt(1 + a**2) * margin
                plt.plot(xx, yy_down, "k--");
                plt.plot(xx, yy_up, "k-.");
            
                # Legend
                if self.legend is True:
                    fig.legend(loc='lower center', fancybox=True, shadow=False, ncol=3,
                        title_fontsize=9, framealpha=0, labelspacing=1, edgecolor='k')#, title="Legend")

                if axes is False:
                    ax.set_xticklabels([])
                    ax.set_xticks([])
                    ax.set_yticklabels([])
                    ax.set_yticks([])
                    #ax.get_xaxis().set_visible(False)
                    #ax.get_yaxis().set_visible(False)
                    plt.gcf().set_facecolor('white')

                if xlabel is not None:
                    ax.set_xlabel(xlabel)
                else:
                    ax.set_xlabel('$x_1$')
                if ylabel is not None:
                    ax.set_ylabel(ylabel)
                else:
                    ax.set_ylabel('$x_2$')
                
                if title is not None:
                    plt.title(title, fontsize=fontsize)

                plt.show();


    #https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html
    def plot_svc_decision_function(self, clf_, i, X1, X2, c1, c2=None, figsize=(12,5), aspect='auto'):
        """Plot the decision function for a 2D SVC"""
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
        ax.set_xlim(self.plt_minx, self.plt_maxx)
        ax.set_ylim(self.plt_miny, self.plt_maxy)

        # Axis limits.
        #x1_min, x1_max = X1.min(), X1.max()
        #x2_min, x2_max = X2.min(), X2.max()
        #ax.set(xlim=(x1_min, x1_max), ylim=(x2_min, x2_max))

        for c, a in zip([c1, c2], [1, -1]):
            t = np.argwhere(self.y[i] == a)
            t = t[:,0]
            if c is None:
                label = 'Rest'
                c = -1
            else:
                label = f'Class {c}'
            
            clr_id = np.where(np.unique(self.y)==a)[0][0]
            ax.scatter(X1[t, self.plt_f1], X2[t, self.plt_f2], s=self.plt_s, color=self.plt_colors[clr_id], label=label);
            #ax.scatter(self.x[i][t,self.plt_f1], self.x[i][t,self.plt_f2], s=50, color=self.plt_colors[int(c)], label=label)#, edgecolor='k', linewidth=1.5)

        # Free support vectors
        #plt.scatter(self.free_sv[i][:, self.plt_f1], self.free_sv[i][:, self.plt_f2], s=90, linewidth=1.8, facecolors="none", edgecolor="dimgray")#, label="Free SV")

        # The points designating the support vectors.
        plt.scatter(self.sv[i][:, self.plt_f1], self.sv[i][:, self.plt_f2], s=self.plt_sv_s, linewidth=self.plt_sv_lw, facecolors="none", edgecolor="k", label="Support vectors")
        
        if self.plt_xx is None or self.plt_yy is None:
            #_X1, _X2 = np.meshgrid(np.linspace(self.plt_minx, self.plt_maxx, 500), np.linspace(self.plt_minx, self.plt_maxx, 500))
            xnum, ynum = plt.gcf().dpi * plt.gcf().get_size_inches()
            xnum, ynum = math.floor(xnum), math.ceil(ynum)
            _X1, _X2 = np.meshgrid(np.linspace(self.plt_minx, self.plt_maxx, num=xnum),
                                    np.linspace(self.plt_miny, self.plt_maxy, num=ynum))
            self.plt_xx = _X1
            self.plt_yy = _X2
        else:
            _X1 = self.plt_xx
            _X2 = self.plt_yy

        xy = np.vstack([_X1.ravel(), _X2.ravel()]).T

        #h = (self.plt_maxx - self.plt_minx)/100
        #_X1, _X2 = np.meshgrid(np.arange(self.plt_minx, self.plt_maxx, h), np.arange(self.plt_miny, self.plt_maxy, h))
        #xy = np.c_[_X1.ravel(), _X2.ravel()]
        # NOTE: a função decision_function retorna um valor por hiperplano (subproblemas binários)
        #P = clf_.decision_function(xy).reshape(_X1.shape)
        Z = clf_.predict(xy).reshape(_X1.shape)
        
        # plot decision boundary and margins
        ax.contour(_X1, _X2, Z, colors='k',
                levels=[-1, 0, 1], alpha=0.5,
                linestyles=['--', '-', '--'])
        
        # plot support vectors
        ax.scatter(clf_.support_vectors_[:, self.plt_f1],
                clf_.support_vectors_[:, self.plt_f2],
                s=300, linewidth=1, facecolors='none')
        
        # Legend
        if self.legend is True:
            fig.legend(loc='lower center', fancybox=True, shadow=False, ncol=3,
                title_fontsize=9, framealpha=0, labelspacing=1, edgecolor='k')#, title="Legend")

        if self.verbose:
            plt.show()
        else:
            """Figures created through the pyplot interface (matplotlib.pyplot.figure) are retained 
            until explicitly closed and may consume too much memory.
            """
            plt.close(fig);


    def plot_svc_subproblems(self, clf, X, y, ith_estimator=None, plot_support=True, colors=None,
                            xlabel=None, ylabel=None, axes=False, title=None, fontsize="small"):

        def plot_svc_decision_function(self, clf, i, X, y, ax=None, plot_support=True, colors=None,
                                        xlabel=None, ylabel=None, axes=False, title=None, fontsize="small"):
            """Plot the decision function for a 2D SVC"""
            if ax is None:
                ax = plt.gca()

            if colors is None:
                colors = ['blue', 'red', 'darkgreen', 'goldenrod', 'cornflowerblue', 'lightcoral', 'springgreen', 
                        'cyan', 'lightsalmon', 'lime', 'darkcyan', 'lightslategray', 'blueviolet', 'magenta',
                        'pink', 'silver', 'black']
            
            #xlim = ax.get_xlim()
            #ylim = ax.get_ylim()
            
            if (self.plt_minx is None or self.plt_maxx is None or
                    self.plt_miny is None or self.plt_maxy is None):
                minx = min(X[:, 0])
                maxx = max(X[:, 0])
                gapx = (maxx - minx) * self.plt_gap_x

                miny = min(X[:, 1])
                maxy = max(X[:, 1])
                gapy = (maxy - miny) * self.plt_gap_y

                minx -= gapx
                miny -= gapy
                maxx += gapx
                maxy += gapy

                self.plt_minx = minx
                self.plt_maxx = maxx
                self.plt_miny = miny
                self.plt_maxy = maxy
            else:
                minx = self.plt_minx
                maxx = self.plt_maxx
                miny = self.plt_miny
                maxy = self.plt_maxy
            
            xlim = (minx, maxx)
            ylim = (miny, maxy)

            #ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn');
            if self.verbose:
                if isinstance(clf, SVC):
                    
                    #svs = clf.support_vectors_[np.where(np.sign(clf.decision_function(clf.support_vectors_))[:, 2] < 0)[0]]
                    #ax.scatter(svs[:, 0], svs[:, 1], s=300, lw=1, facecolors='none', edgecolor="k", label="Support vectors")

                    c1, c2 = [(c1, c2) for (c1, c2) in list(itertools.combinations(np.unique(y), 2) )][i]

                    #for c, a in zip([c1, c2], [1, -1]):
                    for c in [c1, c2]:
                        t = np.argwhere(y == c)
                        t = t[:,0]
                        label = f'Class {c}'

                        #clr_id = np.where(np.unique(y)==c)[0][0]
                        clr_id = np.where(np.unique(self.classes)==c)[0][0]
                        ax.scatter(X[t, 0], X[t, 1], s=self.plt_s, color=colors[clr_id], label=label);
                        #ax.scatter(self.x[i][t,self.plt_f1], self.x[i][t,self.plt_f2], s=50, color=self.plt_colors[int(c)], label=label)#, edgecolor='k', linewidth=1.5)

                    if plot_support:
                        svs_to_plot = np.array([False for _ in range(clf.n_support_.sum())])
                        indices = []
                        # c1
                        for c1_, c2_ in [(c1, c2), (c2, c1)]:
                            print(c1_, c2_)
                            id_c = np.where(clf.classes_ == c1_)[0][0]
                            id_c2 = np.where(clf.classes_ == c2_)[0][0]
                            n_svs = clf._n_support[id_c]

                            if n_svs > 0:

                                if id_c == 0: # não exist dual_coef_ de outras classes antes dessa
                                    id_start = 0
                                    id_end = n_svs
                                else:
                                    # o id_start é igual a soma de svs já descritos para oturas classes em posições anteriores da array
                                    id_start = clf._n_support[:id_c].sum()
                                    id_end = id_start + n_svs

                                indices = [x for x in range(id_start, id_end)]

                                class_in_row = [x for x in range(0, len(np.unique(y)))]
                                #print("class_in_row: ", class_in_row)
                                #class_in_row.remove(c1_) 
                                class_in_row.remove(id_c)
                                #print("class_in_row': ", class_in_row)
                                """
                                    Shape de clf.dual_coef_: (n_classes-1, n_sv)
                                    Para c = 1, a lista 'class_in_row = [0, 2]' indica que:
                                        A primeira linha do clf.dual_coef_ representa o valor obtido para: Classe 1 vs Classe 0
                                        A segunda linha do clf.dual_coef_ representa o valor obtido para: Classe 1 vs Classe 2
                                """
                                #print(class_in_row)
                                #row_to_analyse = class_in_row.index(c2_) # linha do clf.dual_coef_ que se refere às classes 'c' vs 'c2'
                                row_to_analyse = class_in_row.index(id_c2)
                                                            
                                #true_indices = [i for i in indices if np.abs(clf.dual_coef_[row_to_analyse, i]) > 1e-8]
                                true_indices = [i for i in indices if clf.dual_coef_[row_to_analyse, i] != 0]
                                
                                svs_to_plot[true_indices] = True
                        
                        ax.scatter(clf.support_vectors_[:, 0][svs_to_plot], clf.support_vectors_[:, 1][svs_to_plot], s=self.plt_sv_s, lw=1, facecolors='none', edgecolor="k", label="Support vectors")
                else:
                    c1 = clf.classes_[i]
                    c2 = None
                    #for c, a in zip([c1, None], [1, -1]):
                    for c in [c1, c2]:
                        if c is None:
                            t = np.argwhere(y != c1)
                            label = 'Rest'
                            c = -1
                            clr_id = -1
                        else:
                            t = np.argwhere(y == c)
                            label = f'Class {c}'
                            #clr_id = np.where(np.unique(y)==c)[0][0]
                            clr_id = np.where(np.unique(self.classes)==c)[0][0]

                        t = t[:,0]
                        #cmap = LinearSegmentedColormap.from_list('my_list', colors[:len(np.unique(y))])
                        #ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap=cmap);
                        ax.scatter(X[t, 0], X[t, 1], s=self.plt_s, color=colors[clr_id], label=label);

                    if plot_support:
                        ax.scatter(clf.estimators_[i].support_vectors_[:, 0], clf.estimators_[i].support_vectors_[:, 1], s=self.plt_sv_s, lw=1, facecolors='none', edgecolor="k", label="Support vectors")

            # create grid to evaluate model
            if self.plt_xx is None or self.plt_yy is None:
                #x = np.linspace(xlim[0], xlim[1], 30)
                #y = np.linspace(ylim[0], ylim[1], 30)
                #YY, XX = np.meshgrid(y, x)

                xnum, ynum = plt.gcf().dpi * plt.gcf().get_size_inches()
                xnum, ynum = math.floor(xnum), math.ceil(ynum)
                XX, YY = np.meshgrid(np.linspace(self.plt_minx, self.plt_maxx, num=xnum),
                                        np.linspace(self.plt_miny, self.plt_maxy, num=ynum))
                self.plt_xx = XX
                self.plt_yy = YY
            else:
                XX = self.plt_xx
                YY = self.plt_yy

            xy = np.vstack([XX.ravel(), YY.ravel()]).T
            self.plt_xy = xy
            self.clf_xy = clf
            #P = clf.classes_[np.argmax(clf.decision_function(xy), axis=1)].reshape(XX.shape)
            if len(clf.classes_) > 2:
                P = clf.decision_function(xy)[:, i].reshape(XX.shape)
            else:
                P = clf.decision_function(xy).reshape(XX.shape)

            self.margins.append((XX, YY, P))

            # plot decision boundary and margins
            if self.verbose:
                ax.contour(XX, YY, P, colors='k',
                        levels=[-1, 0, 1], alpha=0.5,
                        linestyles=['--', '-', '--'])
            
                # plot support vectors
                #if plot_support:
                #    ax.scatter(clf.support_vectors_[:, 0],
                #                clf.support_vectors_[:, 1],
                #                s=300, linewidth=1, facecolors='none');
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                
                if self.legend is True:
                    ax.legend(loc='lower center', fancybox=True, shadow=False, ncol=3,
                            title_fontsize=9, framealpha=0, labelspacing=1, edgecolor='k',
                            bbox_to_anchor =(0.5, -0.2))#, title="Legend")

                if axes is False:
                    ax.set_xticklabels([])
                    ax.set_xticks([])
                    ax.set_yticklabels([])
                    ax.set_yticks([])
                    #ax.get_xaxis().set_visible(False)
                    #ax.get_yaxis().set_visible(False)
                    plt.gcf().set_facecolor('white')
                    

                if xlabel is not None:
                    ax.set_xlabel(xlabel)
                else:
                    ax.set_xlabel('$x_1$')
                if ylabel is not None:
                    ax.set_ylabel(ylabel)
                else:
                    ax.set_ylabel('$x_2$')
                
                if title is not None:
                    plt.title(title, fontsize=fontsize)

                plt.show()
        
        if ith_estimator is not None:
            plot_svc_decision_function(self, clf, i=ith_estimator, X=X, y=y, plot_support=plot_support, colors=colors,
                                        xlabel=xlabel, ylabel=ylabel, axes=axes, title=title, fontsize=fontsize);
        else:
            print(f"classes_: {clf.classes_}")
            n_classes = len(clf.classes_)
            #if clf.decision_function_shape == 'ovo':
            if isinstance(clf, SVC):
                n_estimators = int(n_classes * (n_classes - 1) / 2)
            elif isinstance(clf, OneVsRestClassifier):
            #elif clf.decision_function_shape == 'ovr':
                if n_classes > 2:
                    n_estimators = n_classes
                else:
                    n_estimators = 1
            else:
                n_estimators = None

            ax=None
            for i in range(n_estimators):
                plot_svc_decision_function(self, clf, i, X=X, y=y, ax=ax, plot_support=plot_support, colors=colors,
                                            xlabel=xlabel, ylabel=ylabel, axes=axes, title=title, fontsize=fontsize);
        
        plt.close()


    def add_point(self, ax, x1, x2, s=50, color=[0.3, 0.3, 0.3], label=None):
        ax.scatter(x1, x2, s=s, color=color, label=label)
        return ax


    def plot_multiclass_boundary(self, point=None, point_label=None, all_margins=False, figsize=(12,5), aspect='auto', 
                                    xlabel=None, ylabel=None, axes=False, title=None, fontsize="small",
                                    cf_alpha=0.25, cf_antialiased=True, cf_kwargs=None,
                                    c_alpha=0.9, c_antialiased=True, c_linestyles='solid', c_linewidths=0,#3,
                                    c_extend='both', legend=True):
        """
        # NOTE: REFERENCE
        - based on the mlxtend library
        https://github.com/rasbt/mlxtend/blob/master/mlxtend/plotting/decision_regions.py
        https://github.com/rasbt/mlxtend/blob/8c61c063f98f0d9646edfd8b0270b77916f0c434/mlxtend/utils/checking.py
        """
        Y_unique = np.unique(self.Y)
        def value_to_index(x):
            # Return position of a given value in the unique array
            return np.where(Y_unique == x)[0][0]
        def vectorize(x):
            return np.vectorize(value_to_index)(x)

        def plot_fig(self, xx, yy, Z, colors, contourf_kwargs, contour_kwargs, 
                        n_classes, all_margins, point=None, point_label=None,
                        figsize=(12,5), aspect='equal',
                        xlabel=None, ylabel=None, axes=False, title=None, fontsize="small",
                        legend=True):

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
                    #if self.kernel  != 'linear':
                    #    ax.contour(margin[0], margin[1], margin[2], [0.0], colors='k', linewidths=2.5, origin='lower')
                    #else:
                    #    ax.plot([self.plt_minx, self.plt_maxx], [margin[0], margin[1]], "k", linewidth=2.5)
                    if self.kernel  != 'linear' or self.plotter != 'plot_margin':
                        ax.contour(margin[0], margin[1], margin[2], [0.0], colors='k', linewidths=2.5, origin='lower');
                    else: #linear and 'plot_margin'
                        ax.plot(margin[0], margin[1], "k");

                    ##for i, (c1, c2) in enumerate(self.superclasses):
                    #for clf in self.superclass_classifiers:
                    #    #clf = self.superclass_classifiers[i]
                    #    for margin in clf.margins:
                    #        if clf.kernel  != 'linear':
                    #            ax.contour(margin[0], margin[1], margin[2], [0.0], colors='k', linewidths=2.5, origin='lower');
                    #        else:
                    #            ax.plot([clf.minx, clf.maxx], [margin[0], margin[1]], "k");
            else:
                OAA_nclasses = self.n_classes - len(self.superclasses)

                if self.OAA_borders:
                    contour = ax.contour(xx, yy, Z, cset.levels, **contour_kwargs)
                    for i, line in enumerate(contour.collections):
                        level_value = contour.levels[i]
                        #if i < OAA_nclasses:
                        if i in self.OAA_borders_lst:
                            print(f"Contour line {i+1}: Level {level_value}")
                            line.set_linestyle(self.OAA_borders_style)  # Change linestyle for the borders
                            line.set_linewidth(self.OAA_borders_lw)  # Adjust the width for emphasis
                else:
                    ax.contour(xx, yy, Z, cset.levels, **contour_kwargs)

            ax.axis([xx.min(), xx.max(), yy.min(), yy.max()])

            #ax = self.add_point(ax, x1=0.4, x2=0.435, label='Point')
            
            #for a in range(0, num_classes):
            for a in list(np.unique(self.Y)):
                t = np.argwhere(self.Y == a)
                t = t[:,0]
                clr_id = np.where(np.unique(self.Y)==a)[0][0]
                ax.scatter(self.X[t,self.plt_f1], self.X[t,self.plt_f2], s=self.plt_s, color=self.plt_colors[clr_id], label=f'Class {a}')#, edgecolor='k', linewidth=1.5)
                
            if point is not None:
                if point_label is None:
                    point_label="Pixel"
                #ax.plot(point[0], point[1], marker='x', markersize=8, color='black')
                ax.scatter(point[0], point[1], s=self.plt_s, color='black', label=point_label)

            # Legend
            if point is not None:
                ncol = n_classes + 1
            else:
                ncol = n_classes

            # Legend
            if legend is True:
                fig.legend(loc='lower center', fancybox=True, shadow=False, ncol=ncol, 
                                title_fontsize=8, framealpha=0, labelspacing=1, edgecolor='k')#, title="Legend")

            if axes is False:
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_yticklabels([])
                ax.set_yticks([])
                #ax.get_xaxis().set_visible(False)
                #ax.get_yaxis().set_visible(False)
                plt.gcf().set_facecolor('white')
                

            if xlabel is not None:
                ax.set_xlabel(xlabel)
            else:
                ax.set_xlabel('$x_1$')
            if ylabel is not None:
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel('$x_2$')
            
            if title is not None:
                plt.title(title, fontsize=fontsize)
                

        if self.dim > 2:
            print("SVM model fitted to more than two features. 2D plots require a change of dimensionality to correctly illustrate the decision boundaries.")
            self.show_dataset()
        else:
            if self.plt_xx is None or self.plt_yy is None:
                #x_min = self.plt_minx
                #x_max = self.plt_maxx
                #y_min = self.plt_miny
                #y_max = self.plt_maxy
                ##dim = self.dim
                #xnum, ynum = plt.gcf().dpi * plt.gcf().get_size_inches()
                #xnum, ynum = math.floor(xnum), math.ceil(ynum)
                ## meshgrid
                #xx, yy = np.meshgrid(np.linspace(x_min, x_max, num=xnum),
                #                    np.linspace(y_min, y_max, num=ynum))

                xnum, ynum = plt.gcf().dpi * plt.gcf().get_size_inches()
                xnum, ynum = math.floor(xnum), math.ceil(ynum)
                xx, yy = np.meshgrid(np.linspace(self.plt_minx, self.plt_maxx, num=xnum),
                                        np.linspace(self.plt_miny, self.plt_maxy, num=ynum))
                self.plt_xx = xx
                self.plt_yy = yy
            else:
                xx = self.plt_xx
                yy = self.plt_yy

            if self.plt_Z is None:
                # Prediction array
                if self.dim == 1:
                    X_predict = np.array([xx.ravel()]).T
                else:
                    X_grid = np.array([xx.ravel(), yy.ravel()]).T
                    #X_predict = np.zeros((X_grid.shape[0], dim))
                    X_predict = np.zeros((X_grid.shape[0], 2))
                    X_predict[:, 0] = X_grid[:, 0]
                    X_predict[:, 1] = X_grid[:, 1]

                # AGG prediction strategy
                Z_ = self.predict(X_predict.astype(self.X.dtype))
                
                #NOTE: VALUES are replaced by their respective position in np.unique(self.Y).
                ## Used to prevent error when Y contains characters/labels instead of integers
                Z_ = vectorize(Z_)

                """
                # Predicted labels
                Z_ = self.predict(X_predict.astype(self.X.dtype), AGG=False)
                
                # Reclassify pixels assigned to superclasses
                #Z_ = np.array(Z_)
                #self.Z_initial = Z_

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
                
                #self.Z_final = Z_
                #Z_ = list(Z_)

                # TODO: only for test
                #Z_ = pd.read_csv('Z_.csv')
                #Z_ = list(Z_.to_numpy().ravel())
                
                #self.Z_ = Z_
                #Z = np.array(Z_).reshape(xx.shape)
                #self.plt_Z = Z
                """

                Z = Z_.reshape(xx.shape)
                self.plt_Z = Z
                
            else:
                Z = self.plt_Z

            n_classes = np.unique(self.Y).shape[0]
            #colors=('#1f77b4,#ff7f0e,#3ca02c,#d62728,'
            #    '#9467bd,#8c564b,#e377c2,'
            #    '#7f7f7f,#bcbd22,#17becf')
            #colors = colors.split(',')
            colors_gen = cycle(self.plt_colors)
            colors = [next(colors_gen) for c in range(n_classes+1)]

            # Plot decision region
            # Make sure contourf_kwargs has backwards compatible defaults
            contourf_kwargs = cf_kwargs#{'extend': 'both', 'corner_mask': True}
            contourf_kwargs_default = {'alpha': cf_alpha, 'antialiased': cf_antialiased}
            contourf_kwargs = format_kwarg_dictionaries(
                                default_kwargs=contourf_kwargs_default,
                                user_kwargs=contourf_kwargs,
                                protected_keys=['colors', 'levels'])

            contour_kwargs = {'alpha': c_alpha, 'antialiased': c_antialiased, 'linestyles':c_linestyles,
                                'linewidths':c_linewidths, 'extend': c_extend}
            contour_kwargs_default = {'linewidths': 0.5, 'colors': 'k', 'antialiased': True}
            contour_kwargs = format_kwarg_dictionaries(
                                default_kwargs=contour_kwargs_default,
                                user_kwargs=contour_kwargs,
                                protected_keys=[])
            
            if all_margins is True:
                plot_fig(self, xx, yy, Z, colors, contourf_kwargs, contour_kwargs, 
                            n_classes, all_margins=all_margins, point=point, point_label=point_label,
                            figsize=figsize, aspect=aspect,
                            xlabel=xlabel, ylabel=ylabel, axes=axes, title=title, fontsize=fontsize,
                            legend=legend)
            
            plot_fig(self, xx, yy, Z, colors, contourf_kwargs, contour_kwargs, 
                            n_classes, all_margins=False, point=point, point_label=point_label,
                            figsize=figsize, aspect=aspect, 
                            xlabel=xlabel, ylabel=ylabel, axes=axes, title=title, fontsize=fontsize,
                            legend=legend)


    def show_dataset(self, multiclass_boundary=False, point=None, point_label=None, figsize=(12,5), aspect='auto', f1=None, f2=None,
                        xlabel=None, ylabel=None, axes=False, title=None, fontsize="small", legend=True):

        if self.X is not None and self.Y is not None:
            # initialize figure
            fig = plt.figure(figsize=figsize)

            gs = gridspec.GridSpec(1, 3, width_ratios=[1,3,1]) 

            # setup current axis
            ax = plt.subplot(gs[1], aspect=aspect); 

            # dress panel
            ax.set_xlim(self.plt_minx, self.plt_maxx)
            ax.set_ylim(self.plt_miny, self.plt_maxy)

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
                clr_id = np.where(np.unique(self.Y)==a)[0][0]
                if f1 is not None and f2 is not None:
                    ax.scatter(self.X[t,f1], self.X[t,f2], s=self.plt_s, color=self.plt_colors[clr_id], label=f'Class {a}')#, edgecolor='k', linewidth=1.5)
                else:
                    ax.scatter(self.X[t,self.plt_f1], self.X[t,self.plt_f2], s=self.plt_s, color=self.plt_colors[clr_id], label=f'Class {a}')#, edgecolor='k', linewidth=1.5)
            
            # Plot support vectors
            ##ax.scatter(self.sv[:, 0], self.sv[:, 1], s=50, linewidth=1.2, facecolors="none", edgecolor="k")#, label="Support vectors")
            #ax.scatter(self.x[self.sv_bool, 0], self.x[self.sv_bool, 1], s=50, linewidth=1.5, facecolors="none", edgecolor="k")#, label="Support vectors")

            if point is not None:
                #ax.plot(point[0], point[1], marker='x', markersize=8, color='black')
                ax.scatter(point[0], point[1], s=self.plt_s, color='black', label=f'Pixel')

            #"""
            if multiclass_boundary is True:
                if self.dim > 2:
                    print("SVM model fitted to more than two features. 2D plots require a change of dimensionality to correctly illustrate the decision boundaries.")
                
                if (f1 is None or f2 is None) or (f1==self.plt_f1 and f2==self.plt_f2):
                    #print(self.margins)
                    for margin in self.margins:
                        if self.kernel  != 'linear' or self.plotter != 'plot_margin':
                            ax.contour(margin[0], margin[1], margin[2], [0.0], colors='k', linewidths=2.5, origin='lower');
                        else: #linear and 'plot_margin'
                            ax.plot(margin[0], margin[1], "k");
                    
                    #for i, (c1, c2) in enumerate(self.superclasses):
                    # for clf in self.superclass_classifiers:
                    #     #clf = self.superclass_classifiers[i]
                    #     for margin in clf.margins:
                    #         #if clf.kernel  != 'linear':
                    #         #    ax.contour(margin[0], margin[1], margin[2], [0.0], colors='k', linewidths=2.5, origin='lower')
                    #         #else:
                    #         #    ax.plot([clf.minx, clf.maxx], [margin[0], margin[1]], "k")
                    #         if clf.kernel  != 'linear' or clf.plotter != 'plot_margin':
                    #             ax.contour(margin[0], margin[1], margin[2], [0.0], colors='k', linewidths=2.5, origin='lower');
                    #         else: #linear and 'plot_margin'
                    #             ax.plot(margin[0], margin[1], "k");
            #"""
            # Legend
            if point is not None:
                ncol = num_classes + 1
            else:
                ncol = num_classes

            if legend is True:
                fig.legend(loc='lower center', fancybox=True, shadow=False, ncol=ncol, 
                    title_fontsize=8, framealpha=0, labelspacing=1, edgecolor='k')#, title="Legend")

            if axes is False:
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_yticklabels([])
                ax.set_yticks([])
                #ax.get_xaxis().set_visible(False)
                #ax.get_yaxis().set_visible(False)
                plt.gcf().set_facecolor('white')

            if xlabel is not None:
                ax.set_xlabel(xlabel)
            else:
                ax.set_xlabel('$x_1$')
            if ylabel is not None:
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel('$x_2$')
            
            if title is not None:
                plt.title(title, fontsize=fontsize)

            plt.show()


        
    def class_separability_analysis(self, method='JM'):

        d = OrderedDict()
        for i, (c1, c2) in enumerate(list(itertools.combinations(self.classes, 2) )):
            dist = self.distances(c1=c1, c2=c2, method=method)

            d[i] = {"pair":i, "c1":c1, "c2":c2, "distance":dist}

        self.separability_method = method

        df_class_separability = pd.DataFrame.from_dict(d, "index")
        self.df_class_separability = df_class_separability
        df_class_separability.sort_values(by=['distance'], ascending=True, inplace=True)
        
        self.class_separability = df_class_separability

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
            try:
                inv_cov = np.linalg.inv(cov)
            except:
                inv_cov = np.linalg.pinv(cov)

            tmp = np.core.dot(m.T, inv_cov)
            tmp = np.core.dot(tmp, m)
            
            MH = np.sqrt(tmp)
            return MH


        def bhattacharyya_distance(m1, m2, cov1, cov2):
            
            m = (m1 - m2)
            cov = (cov1 + cov2)/2
            try:
                inv_cov = np.linalg.inv(cov)
            except:
                inv_cov = np.linalg.pinv(cov)

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
        

    def predict(self, V, AGG_level=None):
        
        def ovo_vote(decision_function, classes, SC_classes):
            combos = list(itertools.combinations(SC_classes, 2))
            votes = np.zeros(len(classes))
            sum_fxs = np.zeros(len(classes))
            for i in range(len(decision_function)):
                if decision_function[i] > 0:
                    votes[ np.where(classes==combos[i][1])[0] ] = votes[ np.where(classes==combos[i][1])[0] ] + 1
                    sum_fxs[ np.where(classes==combos[i][1])[0] ] = sum_fxs[ np.where(classes==combos[i][1])[0] ] + decision_function[i]

                else:
                    votes[ np.where(classes==combos[i][0])[0] ] = votes[ np.where(classes==combos[i][0])[0] ] + 1
                    sum_fxs[ np.where(classes==combos[i][0])[0] ] = sum_fxs[ np.where(classes==combos[i][0])[0] ] + decision_function[i]

            winner = np.argmax(votes)
            
            if votes[winner] > 0 and len(np.where(votes==votes[winner])[0]) == 1:
                return classes[winner]
            else:
                winner_fx = np.argmax(sum_fxs)
                return classes[winner_fx]

        def agg_ovr_vote(decision_function, classes, SC_dict, hiperplanes_classes, last_AGG_clf_classes):
    
            votes = np.zeros(len(classes))
            sum_fxs = np.zeros(len(classes))
            for i in range(len(decision_function)):
                if decision_function[i] > 0:
                    votes[ np.where(classes==hiperplanes_classes[i])[0] ] = votes[ np.where(classes==hiperplanes_classes[i])[0] ] + 1
                    #sum_fxs[ np.where(classes==hiperplanes_classes[i])[0] ] = sum_fxs[ np.where(classes==hiperplanes_classes[i])[0] ] + decision_function[i]
                #else:
                sum_fxs[ np.where(classes==hiperplanes_classes[i])[0] ] = sum_fxs[ np.where(classes==hiperplanes_classes[i])[0] ] + decision_function[i]

            classes_ = classes.copy()
            for superclass_label, original_labels in SC_dict.items():
                for k in original_labels:
                    if k != superclass_label:
                        id_k = np.where(classes_==k)[0]
                        votes[ np.where(classes_==superclass_label)[0] ] = votes[ np.where(classes_==superclass_label)[0] ] + votes[ id_k ]
                        sum_fxs[ np.where(classes_==superclass_label)[0] ] = sum_fxs[ np.where(classes_==superclass_label)[0] ] + sum_fxs[ id_k ]

                        votes = np.delete(votes, id_k)
                        sum_fxs = np.delete(sum_fxs, id_k)
                        classes_ = np.delete(classes_, id_k)

            winner = np.argmax(votes)
            
            if votes[winner] > 0 and len(np.where(votes==votes[winner])[0]) == 1:
                return classes_[winner]

            elif votes[winner] == 0 and len(np.where(votes==votes[winner])[0]) == len(votes) and self.AGG_classify_original is False:
                # NOTE: caso nenhuma classe tenha recebido voto, podemos escolher a classe de saída considerando o maior f(x) obtido.
                # Na abordagem AGG podemos analisar a soma dos valores de f(x) obtidos para cada classe. Porém, é preciso verificar se
                # foi realizado classificação considerando as classes originais, sem combinação de classes, para que a somatória não fique
                # desigual/injusta para a classe mergeada na primeira iteração (terá um valor a menos somado, levando a resultado errado).
                # Nesse cenário, onde 'self.AGG_classify_original is False', escolhemos a classe com o maior f(x) obtido pelo último classificador,
                # o qual apresenta o maior nível de agrupamento de classes.
                #return np.nan
                fxs_last_clf = decision_function[-len(last_AGG_clf_classes):]
                winner_ = np.argmax(fxs_last_clf)
                return last_AGG_clf_classes[winner_]
            
            else:
                # NOTE: caso nenhuma classe tenha recebido voto, podemos escolher a classe de saída considerando o maior f(x) obtido.
                # Nesse caso, onde a classificação inicial considerou as classes originais, podemos escolher a classe com a maior 
                # soma de f(x).
                winner_fx = np.argmax(sum_fxs)
                return classes_[winner_fx]


        if self.strategy == 'OAO' or self.strategy == 'OAA':
            Z = self.clf.predict(V)

        elif self.strategy in ['HAGG', 'AGG', 'mPAGG']:
            # Initial classification/prediction of superclasses
            if self.clf is not None:
                Z = self.clf.predict(V)
            else:
                # The initial classification was not performed
                # All the classes were merged into a unique superclass
                superclass = list(self.superclasses.keys())[0]
                Z = np.full(V.shape[0], superclass)

            # Detailed prediction of superclasses
            for i, (superclass_label, original_labels) in enumerate(self.superclasses.items()):
                #SC_indices = np.where(Z_ == superclass_label)[0]
                SC_indices = np.argwhere(np.isin(Z, [superclass_label])).ravel()
                #SC_indices = np.argwhere(np.isin(Z, original_labels)).ravel()
                V_temp = V[SC_indices]

                SC_clfs_indices = np.where(np.array(self.superclass_per_classifier) == superclass_label)
                clfs = np.array(self.superclass_classifiers)[SC_clfs_indices]

                #d = OrderedDict()
                if len(clfs) > 0:
                    D_list = list()
                    for i, clf in enumerate(clfs):
                        """
                        - decision_function
                        For binary classification, a 1-dimensional array is returned.
                        Values strictly greater than zero indicate the positive class (i.e. the LAST CLASS in classes_).
                        NOTE: Aqui consideramos que os classificadores das superclasses serão sempre binários
                        """
                        fx = clf.decision_function(V_temp)
                        D_list.append(fx)
                        #sign = np.sign(fx)
                        #d[i] = {"classifier":i, "c1":clf.classes_[1], "c2":clf.classes_[0], "fx":fx, "signal":sign, "fx_c2":-fx, "signal_c2":-sign}
                    
                    #df = pd.DataFrame.from_dict(d, "index")

                    D = np.stack(D_list, axis=1)

                    Z_SC = np.apply_along_axis(ovo_vote, 1, D, self.classes, [c for c in original_labels])

                    Z[SC_indices] = Z_SC

        elif self.strategy in ['PAGG']: # REVIEW: refazer a lógica para a abordagem simplificada
            
            # TODO: considerar self.AGG_clfs apenas até o AGG_level, caso este não seja None
            decision_functions = [AGG_clf.clf.decision_function(V) for AGG_clf in self.AGG_clfs]
            
            # NOTE: necessário utilizar reshape para funções de decisões obtidas para apenas 2 classes
            # por serem definidas inicialmente como 1D array, diferente das demais
            D = np.concatenate(
                [d if d.ndim > 1 else np.reshape(d, (-1, 1)) for d in decision_functions],
                axis=1
            )

            last_AGG_clf_classes = self.AGG_clfs[-1].classes
            Z = np.apply_along_axis(agg_ovr_vote, 1, D, self.classes, self.superclasses, [k for c in self.AGG_clfs for k in c.classes], last_AGG_clf_classes) 
            
            # Detailed prediction of superclasses
            for i, (superclass_label, original_labels) in enumerate(self.superclasses.items(), start=1):
                
                # TODO: end loop if i > AGG_level

                #SC_indices = np.where(Z_ == superclass_label)[0]
                SC_indices = np.argwhere(np.isin(Z, original_labels)).ravel()
                if len(SC_indices) > 0:
                    V_temp = V[SC_indices]

                    # NOTE: Alguns procedimentos aqui são desnecessários, já que na AGG cada classe tem apenas um classificador OAA
                    SC_clf_indice = np.where(np.array(self.superclass_per_classifier) == superclass_label)
                    clf = np.array(self.superclass_classifiers)[SC_clf_indice][0]

                    Z_SC = clf.predict(V_temp)

                    Z[SC_indices] = Z_SC

        else:
            raise ValueError(
                "Strategy not found.\n"
                + "The implemented multiclass classification strategies are: OAO, OAA, AGG, PAGG and HAGG"
            )
        
        # fill nan values with the nearest class
        if self.interpolate_nan is True:
            Z = pd.Series(Z).interpolate(method='nearest').to_numpy()

        return Z


    def classify(self, dataset, classes=None, test_dataset=None, test_expected_arr=None, AGG_level=None):
        
        start = time.time()

        ## Para OAA-SVM com tie_break=True, argmax(clf.decision_function(X), axis=1) é igual a clf.predict(X)
        #np.argmax(clf.decision_function(X), axis=1)
        #clf.predict(X)

        #def predict(self, X):
        #    D = self.decision_function(X)
        #    return self.classes_[np.argmax(D, axis=1)]
        #
        #self.predict(X)

        def accuracy_assessment(expected_arr, predicted_arr, classes=None):
            OA_test = round(accuracy_score(expected_arr, predicted_arr), 4)
            kappa_test = round(cohen_kappa_score(expected_arr, predicted_arr), 3)

            # confusion matrix for the testing dataset
            confusion_pred = pd.crosstab(expected_arr, predicted_arr, rownames=['Actual'], colnames=['Predicted'], margins=True, margins_name="Total")
            if classes is None:
                expected_rows = list(pd.unique(expected_arr))
            else:
                expected_rows = list(classes.keys())
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

            if classes is not None:
                confusion_pred = confusion_pred.rename(columns=classes, index=classes).fillna('')
            return OA_test, kappa_test, confusion_pred

        if dataset is not None:
            if type(dataset) != np.ndarray:
                raise ValueError(
                    "Dataset type not accepted."
                    + "The dataset parameter should be a numpy array."
                )

        if test_dataset is None or test_expected_arr is None:
            OA_test = None
            kappa_test = None
            confusion_pred = None
        elif type(test_dataset) != np.ndarray:
            raise ValueError(
                "Dataset type not accepted for the test data."
                + "The test_dataset parameter should be a numpy array."
            )

        if dataset is not None:
            predicted_arr = self.predict(dataset, AGG_level=AGG_level)
        else: 
            predicted_arr = None

        if test_dataset is not None and test_expected_arr is not None:
            test_predicted_arr = self.predict(test_dataset, AGG_level=AGG_level)
            OA_test, kappa_test, confusion_pred = accuracy_assessment(test_expected_arr, test_predicted_arr, classes=classes)

        accuracy_dict = {'OA': OA_test, 'Kappa': kappa_test, 'confusion_matrix': confusion_pred}
        
        #if test_mask_arr is not None:
        #    
        #    for c, class_ in self.classes:
        #        indexes = np.where(test_mask_arr==c)
        #        if (indexes[0].size > 0):

        self.time['classify'] = time.time() - start

        return predicted_arr, accuracy_dict