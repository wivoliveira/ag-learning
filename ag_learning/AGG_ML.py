import math
import time
import warnings
import itertools
import numpy as np
import pandas as pd

from copy import deepcopy
from scipy.stats import mode
from collections import OrderedDict
from sklearnex import patch_sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, cohen_kappa_score
patch_sklearn()


# from .class_separability import class_separability_analysis


class MultivariateGaussianMLC:

    def __init__(self, equiprobable_classes=True, use_principal_components=False) -> None:
        self.equiprobable_classes = equiprobable_classes
        self.use_principal_components = use_principal_components
        self.class_mean = None
        self.class_cov = None
        self.classes = None

    def fit(self, X_train, y_train):
        self.class_mean = []
        self.class_cov = []
        self.classes = np.unique(y_train)

        # BASIC PARAMS
        self.class_mean = [
            np.array(np.mean(X_train[y_train==c], axis=0)) 
            for c in self.classes
        ]
        self.class_cov = [
            np.cov(X_train[y_train==c], rowvar=False) 
            for c in self.classes
        ]

    def predict(self, X):
    
        # Multivariate Gaussian Probability Density Function, using Principal Components
        def PC_multivariate_gaussian(x, mu, sigma, class_prob=1.0):
            
            def principal_components(mean, _cov):
                (evals, evecs) = np.linalg.eigh(_cov)
                # numpy says eigenvalues may not be sorted so we'll sort them.
                ii = list(reversed(np.argsort(evals)))
                evals = evals[ii]
                evecs = evecs[:, ii]
                #_pcs = PrincipalComponents(evals, evecs, mean)
                
                #transform = LinearTransform(evecs.T, pre=-mean)

                return {'eigenvalues':evals, 'eigenvectors': evecs}#, 'transform': transform}

            pcs = principal_components(mu, sigma)

            #n = mu.shape[0]
            delta = (x - mu)
            sigma_inv = np.linalg.inv(sigma)
            log_det_cov = np.sum(np.log([v for v in pcs['eigenvalues'] if v > 0]))
            
            likelihood = math.log(class_prob) - 0.5 * log_det_cov \
                            - 0.5 * ((delta).T).dot(sigma_inv).dot(delta)

            return likelihood

         # Multivariate Gaussian Probability Density Function
        def multivariate_gaussian(x, mu, sigma):
            n = mu.shape[0]
            sigma_det = np.linalg.det(sigma)
            #(sign, logdet) = np.linalg.slogdet(sigma)
            sigma_inv = np.linalg.inv(sigma)
            factor = 1/np.sqrt((2 * np.pi)**n * sigma_det)
            exp = (((x - mu).T).dot(sigma_inv)).dot(x - mu)

            return np.exp(-0.5 * exp)/factor

        # Multivariate Gaussian ML discriminant Function for equiprobable classes
        def multivariate_gaussian_equiprobable(x, mu, sigma):
            
            #(sign, logdet) = np.linalg.slogdet(sigma)
            sigma_det = np.linalg.det(sigma)
            sigma_inv = np.linalg.inv(sigma)

            return -np.log(sigma_det) - (((x - mu).T).dot(sigma_inv)).dot(x - mu)

        classes_idx_nd = [[x] for x in np.arange(len(self.classes))]
        if self.equiprobable_classes:
            if self.use_principal_components:
                ML_likelihoods = np.apply_along_axis(
                    lambda y1: np.apply_along_axis(
                        lambda x1: PC_multivariate_gaussian(x1, self.class_mean[int(y1)], self.class_cov[int(y1)]),
                        1, X), 1, classes_idx_nd).T
            else:
                ML_likelihoods = np.apply_along_axis(
                    lambda y1: np.apply_along_axis(
                        lambda x1: multivariate_gaussian_equiprobable(x1, self.class_mean[int(y1)],
                                                                      self.class_cov[int(y1)]),
                        1, X), 1, classes_idx_nd).T
        else:
            ML_likelihoods = np.apply_along_axis(
                lambda y1: np.apply_along_axis(
                    lambda x1: multivariate_gaussian(x1, self.class_mean[int(y1)], self.class_cov[int(y1)]),
                    1, X), 1, classes_idx_nd).T

        ML_idxs = np.apply_along_axis(lambda x1: np.argmax(x1), 1, ML_likelihoods)
        ML_labels = list(map(lambda x: self.classes[int(x)], ML_idxs))

        df_likelihoods = pd.DataFrame(ML_likelihoods, columns=self.classes)
        
        return np.array(ML_labels)#, df_likelihoods


class AGG_ML:

    def __init__(self,
        classifier='GaussianNB', strategy='HAGG', HAGG_N = None,
        AGG_default_modell = 'GaussianNB',
        AGG_classify_original = True, 
        distance_restriction = False, scale_data=False, interpolate_nan=False, 
        distance_method='JM', AGG_dist_threshold=1.5,
        classes_dict = {},

        tol=1e-3, tunning_OA_threshold=1,

        equiprobable_classes=False, 
        use_principal_components=True,

        verbose=2, random_state=10
    ) -> None:

        warnings.filterwarnings("ignore", message="No contour levels were found within the data range.")
        warnings.simplefilter(action='ignore', category=FutureWarning)

        self.verbose = verbose
        self.random_state = random_state
        
        # CLASSIFIERS
        self.classifier = classifier
        self.strategy = strategy

        self.classes_dict = classes_dict
        self.classes = []
        self.classes_interest = []
        self.classes_antagonists = []

        self.clf = None
        self.AGG_clfs = []
        self.clf_xy = None       

        self.dim = None
        self.class_mean = []
        self.class_cov = []

        self.n_classes = None
        self.AGG_n_classes = None
        self.distance_method = distance_method
        self.AGG_dist_threshold = AGG_dist_threshold
        self.AGG_default_modell = AGG_default_modell

        # WORKFLOW
        self.HAGG_N = HAGG_N
        self.AGG_classify_original = AGG_classify_original
        self.scale_data = scale_data
        self.interpolate_nan = interpolate_nan
        self.distance_restriction = distance_restriction

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

        # list of merged classes [(a,b), (c,e)]
        self.superclasses = dict()
        self.superclass_tests = []
        self.superclass_tests_summary = []
        self.superclass_classifiers = []
        self.superclass_per_classifier = []
        self.superclass_best_params = []
        
        self.equiprobable_classes = equiprobable_classes
        self.use_principal_components = use_principal_components

        self.extra_params = dict()


    def get_data_params(self):
        
        # BASIC PARAMS
        self.class_mean = [np.array(np.mean(self.X[self.Y==c], axis=0)) for c in self.classes]
        self.class_cov = [np.cov(self.X[self.Y==c], rowvar=False) for c in self.classes]


    def get_classification_model(self):
        
        #print("get_classification_model() -> ", self.classifier)

        if self.classifier == 'GaussianNB':
            model = GaussianNB()
            
        elif self.classifier == 'MultivariateGaussianMLC':
            model = MultivariateGaussianMLC(
                equiprobable_classes = self.equiprobable_classes, 
                use_principal_components = self.use_principal_components
            )
        else:
            raise ValueError(
                "Classifier not found.\n"
                + "The implemented classifiers are: GaussianNB, MultivariateGaussianMLC"
            )
        return model
        

    def fit(self, X, y, classes_interest=[]):

        start = time.time()

        self.X = X
        self.Y = y
        self.dim = X.shape[1]
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)

        if bool(classes_interest):
            self.classes_interest = [
                x for x in classes_interest 
                if x in self.classes
            ]

        self.classes_antagonists = [
            x for x in self.classes 
            if x not in self.classes_interest
        ]

        self.get_data_params()
        
        if self.strategy == 'antagonistic':
            self.antagonistic_learning(X, y)
        else:
            model = self.get_classification_model()
            model.fit(X, y)
            self.clf = model
        
        self.time['fit'] = time.time() - start
    

    def antagonistic_learning(self, X, y):
        n_clfs = 0
        ## --- CLASS SEPARABILITY ANALYSIS ---
        df_class_separability = self.class_separability_analysis(method=self.distance_method)

        self.AGG_class_separability_original = df_class_separability
        
        if self.classes_interest:

            df_class_separability = df_class_separability.loc[
                ~(
                    df_class_separability['c1'].isin(self.classes_interest) 
                    | df_class_separability['c2'].isin(self.classes_interest)
                )
            ]

        df_class_separability, classes_remap = self.filter_separability_dataframe(df_class_separability, two_class_clusters=False, distance_restriction=self.distance_restriction)
        self.classes_remap = classes_remap

        model = self.get_classification_model()

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
        #print("superclasses ", superclasses)

        y_ = y.copy()

        self.superclasses = superclasses

        # Agrupando as superclasses primeiramente
        for sc, sc_classes in superclasses.items():
            if self.verbose is not False:
                print(sc, sc_classes)
            #self.superclasses[sc] = np.array(sc_classes)

            SC_indices = np.argwhere(np.isin(y, sc_classes)).ravel()
            y_[SC_indices] = sc

        AGG_unique_labels = np.unique(y_)

        if len(AGG_unique_labels) > 1:
            n_clfs = n_clfs + 1
            #print(f"clf{n_clfs} - {self.AGG_default_strategy} (merged)")

            clf_model_ = deepcopy(model)
            clf_model_.fit(X, y_)
            self.clf = clf_model_
            #self.AGG_clfs.append(clf_model_) #TODO

        self.x = [] #TODO
        self.y = [] #TODO

        # # REVIEW: turn to a binary classifier, if necessary
        # for sc, sc_classes in superclasses.items():
        #     print(f"Superclass: {sc} -> {sc_classes}")
        #     sc_indices = np.argwhere(np.isin(y, sc_classes)).ravel()
            
        #     X_sc = X[sc_indices]
        #     y_sc = y[sc_indices]
            
        #     # TODO: call parameter optimization procedure, if it applies
        #     #--
        #     #--

        #     clf = deepcopy(model)
        #     clf.fit(X_sc, y_sc)

        #     self.superclass_classifiers.append(clf)
        #     self.superclass_per_classifier.append(sc)
            

        # Binary classification of superclasses
        for sc, sc_classes in superclasses.items():
            #print(f"Superclass: {sc} -> {sc_classes}")
            
            for i, (c1, c2) in enumerate(list(itertools.combinations(sc_classes, 2) )):
                n_clfs = n_clfs + 1

                c1_c2_indices = np.argwhere(np.isin(y, [c1, c2])).ravel()
            
                X_binary = X[c1_c2_indices]
                y_binary = y[c1_c2_indices]
            
                # TODO: call parameter optimization procedure, if it applies
                #--
                #--

                clf = deepcopy(model)
                clf.fit(X_binary, y_binary)

                self.superclass_classifiers.append(clf)
                self.superclass_per_classifier.append(sc)



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


    def class_separability_analysis(self, method='JM'):

        d = OrderedDict()
        for i, (c1, c2) in enumerate(list(itertools.combinations(self.classes, 2) )):
            dist = self.distances(c1=c1, c2=c2, method=method)

            d[i] = {"pair":i, "c1":c1, "c2":c2, "distance":dist}

        self.separability_method = method

        df_class_separability = pd.DataFrame.from_dict(d, "index")
        self.df_class_separability = df_class_separability
        df_class_separability.sort_values(by=['distance'], ascending=True, inplace=True)
        
        if bool(self.classes_dict):
            df_class_separability['c1_class'] = df_class_separability.c1.map(self.classes_dict)
            df_class_separability['c2_class'] = df_class_separability.c2.map(self.classes_dict)

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
        

    def predict(self, V):
        
        #print(f"predict() ->  {self.strategy}")

        if self.strategy == 'antagonistic':
            #print("antagonistic strategy")
            # Initial classification/prediction of superclasses
            if self.clf is not None:
                #print("self.clf is not None")
                Z = self.clf.predict(V)
                #print(Z.shape)
            else:
                #print("self.clf is None")
                # The initial classification was not performed
                # All the classes were merged into a unique superclass
                superclass = list(self.superclasses.keys())[0]
                Z = np.full(V.shape[0], superclass)
                #print(Z.shape)

            # Detailed prediction of superclasses
            for i, (superclass_label, original_labels) in enumerate(self.superclasses.items()):
                #print(i, superclass_label, original_labels)
                #SC_indices = np.where(Z_ == superclass_label)[0]
                SC_indices = np.argwhere(np.isin(Z, [superclass_label])).ravel()
                #SC_indices = np.argwhere(np.isin(Z, original_labels)).ravel()
                V_temp = V[SC_indices]

                SC_clfs_indices = np.where(np.array(self.superclass_per_classifier) == superclass_label)
                clfs = np.array(self.superclass_classifiers)[SC_clfs_indices]

                #print(f"len(clfs): {len(clfs)}")
                if len(clfs) > 0:
                    D_list = list()
                    for i, clf in enumerate(clfs):
                        y_ = clf.predict(V_temp)
                        D_list.append(y_)
                        
                    D = np.stack(D_list, axis=1)

                    result, _ = mode(D, axis=1)
                    Z_SC = result.flatten()

                    Z[SC_indices] = Z_SC

        else:
            #print("NOT antagonistic strategy")
            Z = self.clf.predict(V)

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
            predicted_arr = self.predict(dataset)
        else: 
            predicted_arr = None

        if test_dataset is not None and test_expected_arr is not None:
            test_predicted_arr = self.predict(test_dataset)
            OA_test, kappa_test, confusion_pred = accuracy_assessment(test_expected_arr, test_predicted_arr, classes=classes)

        accuracy_dict = {'OA': OA_test, 'Kappa': kappa_test, 'confusion_matrix': confusion_pred}
        
        #if test_mask_arr is not None:
        #    
        #    for c, class_ in self.classes:
        #        indexes = np.where(test_mask_arr==c)
        #        if (indexes[0].size > 0):

        self.time['classify'] = time.time() - start

        return predicted_arr, accuracy_dict