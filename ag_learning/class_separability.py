import itertools
import numpy as np
import pandas as pd
from collections import OrderedDict
    
    
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
        """
        In many software implementations (such as Matlab or 
        QGIS Semi-Automatic Classification Plugin) the 
        Jeffries-Matusita distance is calculated as 
                    2 * (1 - exp(-Bhat)) 
        instead of 
                sqrt(2 * (1 - exp(-Bhat))). 
        Therefore, the range of the variable is [0, 2]. 
        It is important to define which formula has been used 
        in order to be able to compare the research results.
        """
        
        #JM = np.sqrt(2 * (1 - np.exp(-B)))
        JM = 2 * (1 - np.exp(-B))
        return JM

    classes = df[label_column].unique()

    if c1 not in classes or c2 not in classes:
        raise ValueError(
            "Both c1 and c2 must be amongst the object classes."
        )

    x_c1 = df.loc[df[label_column]==c1, df.columns!=label_column]
    x_c2 = df.loc[df[label_column]==c2, df.columns!=label_column]

    mean_c1 = np.array(np.mean(x_c1))
    mean_c2 = np.array(np.mean(x_c2))

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


def class_separability_analysis(df, label_column, method='JM', classes_dict={}):

    d = OrderedDict()
    for i, (c1, c2) in enumerate(list(itertools.combinations(df[label_column].unique(), 2) )):
        dist = distances(df, label_column, c1=c1, c2=c2, method=method)

        d[i] = {"pair":i, "c1":c1, "c2":c2, "distance":dist}

    df_class_separability = pd.DataFrame.from_dict(d, "index")
    df_class_separability.sort_values(by=['distance'], ascending=True, inplace=True)
    
    if bool(classes_dict):
        df_class_separability['c1_class'] = df_class_separability.c1.map(classes_dict)
        df_class_separability['c2_class'] = df_class_separability.c2.map(classes_dict)

    return df_class_separability


def filter_separability_dataframe(df_class_separability, AGG_dist_threshold=1.5, two_class_clusters=True, distance_restriction=False):
        
    classes_remap = {}
    num_superclasses = 0
    
    if AGG_dist_threshold is not None:
        df_class_separability = df_class_separability.loc[
            df_class_separability['distance'] <= AGG_dist_threshold
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