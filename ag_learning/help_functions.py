
import os
import math
import psutil
import numpy as np
import pandas as pd
from osgeo import gdal
from numpy.linalg import eig

def usage():
    process = psutil.Process(os.getpid())
    return process.memory_info()[0] / float(2 ** 20)


def acc_euclidian_distance(p1, p2, opt_value=1, sn=3):
    """
    Euclidian Distance between two points, where the second points indicates the optimum accuracy values
    p1 and p2 indicates the pair of accuracies values that compose the first point
    P1 = (p1, p2) = (0.821, 0.553)
    P2 = (opt_value, opt_value) = (1, 1)

    ED = sqrt( (0.821 - 1)^2 + (0.553 - 1)^2 ) = 0.482
    """
    return round(
        math.sqrt((p1 - opt_value)**2 + (p2 - opt_value)**2),
        sn
    )

def Bell_partitions(set_):
    if not set_:
        yield []
        return
    for i in range(int(2**len(set_)/2)):
        parts = [set(), set()]
        for item in set_:
            parts[i&1].add(item)
            i >>= 1
        for b in Bell_partitions(parts[1]):
            yield [parts[0]]+b
            
def sort_list_sets(list_):
    #list_.sort()
    list_ = sorted(list_, key=lambda d: list(d)[0]) 
    return list_


def read_GeoTiff(file_dir):
    img = gdal.Open(file_dir)

    img_Nrows = img.RasterYSize
    img_Ncols = img.RasterXSize
    img_Nbands = img.RasterCount
    img_GeoTransform = img.GetGeoTransform()
    img_Projection = img.GetProjection()

    img_arr = img.ReadAsArray(0, 0, img_Ncols, img_Nrows) # xoff, yoff, xcount, ycount
    img = None

    return img_arr, img_Nrows, img_Ncols, img_Nbands, img_GeoTransform, img_Projection


# Write a classification map to GeoTIFF file
def Write_GeoTiff(array, filename, Nrows, Ncols, Nbands, geotransform=None, projection=None):
    driver = gdal.GetDriverByName('GTiff')
    
    dataset_output = driver.Create(filename, Ncols, Nrows, Nbands, gdal.GDT_Int32)#gdal.GDT_Float32)
    for b in range(0, Nbands):
        dataset_output.GetRasterBand(b+1).WriteArray(array[b])
    
    if geotransform is not None:
        gt = list(geotransform)
        dataset_output.SetGeoTransform(tuple(gt))
    
    if projection is not None:        
        dataset_output.SetProjection(projection)
    
    dataset_output = None


def write_samples_to_file(df, dir_output, filename, target_col=None, target=None):

    #Include the class column in the dataframes
    if target_col is not None and target is not None:
        df[target_col] = target

    filepath = os.path.join(dir_output, filename)

    if not os.path.exists(dir_output):
        # NOTE makedirs is able to create folders and subfolders
        #os.mkdir(dir_output)
        os.makedirs(dir_output)
    
    # Writing sample data sets to files
    try:
        pd.DataFrame(df).to_csv(filepath, index=False, header=True)
    except Exception as e:
        print(str(e))


# Extract samples from a mask of samples
def extract_samples_from_mask(img_arr, mask_arr, classes, bands, class_col='target'):
    #samples = []
    n_samples_class = []
    #classes_found = []

    df_indices = pd.DataFrame(np.hstack((np.indices(mask_arr.shape).reshape(2, mask_arr.size).T,\
                        mask_arr.reshape(-1, 1))), columns=['row', 'col', 'value'])
    df_indices.reset_index(inplace=True)
    
    Ncols = mask_arr.shape[1]

    df_samples_all = pd.DataFrame(np.nan, index=range(0), columns=bands+[class_col])
    df_samples_all = df_samples_all.rename_axis(index='index', axis=0)
    for c, class_ in classes.items():
        indexes = np.where(mask_arr==c)
        
        if (indexes[0].size > 0):
            #classes_found.append(c)
            n_samples_class.append(indexes[0].size)
            df_temp = pd.DataFrame(np.nan, index=range(indexes[0].size), columns=bands+[class_col])
            
            df_temp[class_col] = c
            for b, band in enumerate(bands):
                a = img_arr[b, indexes[0], indexes[1]]
                df_temp[band] = pd.DataFrame(a)
            
            df_temp['index'] = pd.DataFrame((Ncols*indexes[0])+indexes[1])
            df_temp = df_temp.set_index(['index'])
            
            df_samples_all = pd.concat([df_samples_all, df_temp])
            #samples.append(df_temp)
        else:
            n_samples_class.append(0)
            #df_temp = pd.DataFrame(columns=bands+[class_col])
            #df_temp = pd.DataFrame(np.nan, index=range(1), columns=bands+[class_col])
            #df_temp[class_col] = c
            #samples.append(df_temp)

    return df_samples_all, n_samples_class
#df_samples_all, n_samples_class = extract_samples_from_mask(img_arr, GT_arr, class_dict, bands, class_col='target')


def rect_synthetic_img_generator(Nrows, Ncols, bands, classes, df_samples, target_col, vertical_splits, 
                                    fill_empty=True, adjust_block_row=True, adjust_first_block_row=True, simulate_values=True, seed=1,
                                    mean_dt=(0.90, 1.10), cov_dt=(0.55, 1.45), classes_dt=[]):

    synthetic_structure = np.zeros((Nrows, Ncols), dtype=int)
    n_classes = len(np.unique(classes))

    if not isinstance(vertical_splits, int):
        vertical_splits = int(vertical_splits)

    if vertical_splits > n_classes:
        vertical_splits = n_classes

    # Number of classes/blocks in the last block row
    rest = n_classes%vertical_splits
    if rest == 0:
        # Number of classes perfectly split into n vertical blocks
        horizontal_splits = int(n_classes/vertical_splits)
    elif rest == 1:
        # If we use the requested number of vertical splits, the last row will have only 1 blocks
        # if adjust_block_row is True, the 'block row' before the last will be squeezed to include one more block. The 'height' of the block rows will be adjusted
        # if adjust_block_row is False, the last 'block row' will have only one class, we might be extended to fill the block row, if fill_empty is True 
        if adjust_block_row:
            # number of vertical splits for the first/last row block
            vertical_splits_2 = vertical_splits+1
            # block width for the first/last row block
            block_size_cols_2 = int(Ncols/vertical_splits_2)
            horizontal_splits = int(n_classes/vertical_splits) #int() ensures the correct number of row blocks
        else:
            horizontal_splits = math.ceil(n_classes/vertical_splits)
            #if fill_empty: # extend block width
            #else: # preserve block width and background value (0)
    else: #rest > 1:
        # The last 'block row' contain more than 1 block (between 2 and (vertical_splits-1) blocks),
        # if adjust_block_row is True, adjust the widths of the blocks
        if adjust_block_row:
            # number of vertical splits for the first/last row block
            vertical_splits_2 = rest
            block_size_cols_2 = int(Ncols/rest)
        horizontal_splits = math.ceil(n_classes/vertical_splits)

    # block width for all block rows except the first/last (according to adjust_first_block_row)
    block_size_cols = int(Ncols/vertical_splits)

    # block height for all block rows
    block_size_rows = int(Nrows/horizontal_splits)

    if adjust_first_block_row:
        adjust_row_id = 0
    else:
        adjust_row_id = horizontal_splits-1

    c = 0
    for hs in range(0, horizontal_splits):
        col_start = 0 
        row_start = (hs * block_size_rows)
        row_finish = int(row_start + block_size_rows)

        if rest != 0 and hs == adjust_row_id:
            row_vertical_splits = vertical_splits_2
        else:
            row_vertical_splits = vertical_splits
            
        for vs in range(0, row_vertical_splits):

            if rest == 0 or hs != adjust_row_id or not adjust_block_row: # if perfect split or not the block row to be adjusted or not to adjust block widths
                col_start = (vs * block_size_cols)
                col_finish = int(col_start + block_size_cols)
            else:#if rest > 0 and adjust_block_row:
                col_start = (vs * block_size_cols_2)
                col_finish = int(col_start + block_size_cols_2)
            
            synthetic_structure[row_start:row_finish, col_start:col_finish] = classes[c]
            
            # In case border pixels remain unclassified
            if (vs+1) == row_vertical_splits and fill_empty:
                synthetic_structure[row_start:row_finish, col_finish:] = classes[c]

            c = c + 1

    # NOTE Reshape the array to (Npixels, 1), column by column
    synthetic_structure_T = synthetic_structure.T.reshape(Nrows*Ncols, 1, order='F')

    # NOTE Creating N-dimensional raster using the structure defined above
    synthetic_img = np.zeros((Nrows*Ncols, len(bands)), dtype=int)
    #synthetic_img = np.zeros((Nbands, Nrows, Ncols), dtype=int)
    # NOTE Reshape the array to (Npixels, 1), column by column
    #synthetic_img = synthetic_img.T.reshape(Nrows*Ncols, Nbands, order='F') # reshapes column by column
    
    # if simulate_values is False just select random samples to fill the image
    # else extract statistics from the samples and simulate class values
    if not simulate_values: 
        print("Selecionando pixels aleatórios para popular a imagem")
        # Random samples
        # NOTE: This number of pixels will not be enough if the classes on the right contain a larger number of pixels
        try:
            n_random_pixels = math.ceil((Nrows*Ncols*1.2)/len(classes)) 
        except:
            n_random_pixels = math.ceil((Nrows*Ncols)/len(classes)) 

        dfs = []
        #for class_ in range(1, Nclasses+1):
        for class_ in classes:
            dfs.append(df_samples.loc[df_samples[target_col]==class_, df_samples.columns != target_col].sample(n_random_pixels, random_state=seed))

        # NOTE Fill the synthetic image
        for id, class_ in enumerate(classes):
            i = 0
            for pixel in range(len(synthetic_img)):
                pixel_class = synthetic_structure_T[pixel][0]
                #print(f"Classe analisada: {class_}, Pixel {pixel}, classe {pixel_class}")
                if pixel_class == class_:
                    #print(np.array(dfs[id].iloc[i])[:Nbands])
                    synthetic_img[pixel] = dfs[id].iloc[i][bands]
                    i = i + 1

    else:  
        print("Simulando pixels a partir de estatísticas extraídas de cada classe")
        np.random.seed(seed)

        df_samples = df_samples.loc[:, bands+[target_col]]

        # mean vector
        mean_vector = df_samples[df_samples[target_col].isin(classes)].groupby(by=[target_col]).mean()

        # covariance matrix
        class_cov = df_samples[df_samples[target_col].isin(classes)].groupby(by=[target_col]).cov()

        #eigen_vals = []
        eigen_vecs = []
        eigen_vals_sqrt = []

        # NOTE Fill the synthetic image
        for id, class_ in enumerate(classes):
            
            # NOTE: only required if each class block is divided into distinct segments/regions
            # random scalars uniformly distributed, used to model the mean and variance intraclass fluctutions
            segments = 1
            segm = 0 
            if class_ in classes_dt:
                mean_fluctuations = np.random.uniform(low=mean_dt[0], high=mean_dt[1], size=segments) #mean_dt = (0.80, 1.20)
                cov_fluctuations = np.random.uniform(low=cov_dt[0], high=cov_dt[1], size=segments) # #cov_dt = (0.45, 1.55)
            else:
                mean_fluctuations = np.random.uniform(low=0.9, high=1.1, size=segments) #mean_dt = (0.80, 1.20)
                cov_fluctuations = np.random.uniform(low=0.55, high=1.45, size=segments) # #cov_dt = (0.45, 1.55)

            vals, vecs = eig(class_cov.loc[[class_]])
            #eigen_vals.append(vals)
            eigen_vals_sqrt.append(np.sqrt(vals))
            eigen_vecs.append(vecs)

            for pixel in range(len(synthetic_img)):
                pixel_class = synthetic_structure_T[pixel][0]
                #print(f"Classe analisada: {class_}, Pixel {pixel}, classe {pixel_class}")
                if pixel_class == class_:
                    # a 1-dimensional random vector generated by a standard Multivariate Gaussian distribution
                    """
                        np.random.randn generates samples from the normal distribution (mean = 0 and variance = 1)
                        np.random.rand generates samples from a uniform distribution (in the range [0,1))
                        np.random.normal draws random samples from a normal (Gaussian) distribution (loc: mean, scale: std)
                        no.random.standard_normal draws samples from a standard Normal distribution (mean=0, stdev=1).
                        np.random.multivariate_normal draws random samples from a multivariate normal distribution (mean: 1xN array, cov: NxN array).
                    """
                    #random_vect = [-1]
                    #while len([v for v in random_vect if v < 0]) > 0:
                    ##random_vect = np.random.multivariate_normal(mean_vector.loc[[class_]].to_numpy()[0], class_cov.loc[[class_]], size=mean_vector.shape[1])[0]  
                    ##random_vect = np.random.normal(loc=0, scale=0.5, size=mean_vector.shape[1])
                    random_vect = np.random.randn(1, mean_vector.shape[1])[0]

                    # matrix multiplication: np.dot(A, B)
                    # pixel_values = ((eigen_vecs X eigen_vals_sqrt X random_vect) X segm_cov_scalar) + (mean_vector X segm_mean_scalar)

                    pixel_vector = np.dot(np.dot(np.dot(eigen_vecs[id], eigen_vals_sqrt[id]), random_vect), cov_fluctuations[segm]) + np.dot(mean_vector.loc[[class_]].to_numpy()[0], mean_fluctuations[segm])
                    pixel_vector[pixel_vector < 0] = 0
                    synthetic_img[pixel] = pixel_vector

    synthetic_img_reshaped = synthetic_img.T.reshape(len(bands), Nrows, Ncols)
    synthetic_structure_reshaped = synthetic_structure_T.reshape(1, Nrows, Ncols)

    return synthetic_img, synthetic_structure_T, synthetic_img_reshaped, synthetic_structure_reshaped


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