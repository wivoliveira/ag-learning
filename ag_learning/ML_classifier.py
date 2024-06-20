import math
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings(action='ignore')


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

        