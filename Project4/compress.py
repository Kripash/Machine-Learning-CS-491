import os
import matplotlib.pyplot as plt
import numpy as np
import pca

def load_data(input_dir):
    if os.path.exists(input_dir):
        flattened_images = []
        for image in os.listdir(input_dir):
            flattened_images.append( plt.imread(input_dir + '/' + image).flatten() )

        flattened_images = np.transpose(flattened_images)
        flattened_images = flattened_images.astype(float) ## convert all values in 
        # flattened_images to float
        return flattened_images

def compress_images(DATA, k):
    DATA = np.transpose(DATA)

    Z = pca.compute_Z(DATA, centering=True, scaling=True)
    cov_z = pca.compute_covariance_matrix(Z)
    L, PCS = pca.find_pcs(cov_z)
    Z_star = pca.project_data(Z, PCS, L, k, 0)
    U_T = np.transpose(PCS) ## Problem starts here, we need only k pc's
    print('\n\n\n\n\n\n\n\n\n\n\n\n')
    # At this point, PCS is of size 2880x2880 since there are 2880 'features' (our dimension)
    # and so it is assumed that there will be 2880 eigenvalues, and hence 2880 respective 
    # eigenvectors. So, U_T is also 2880 x 2880
    compressed_images = np.dot(np.transpose(Z_star), U_T)
    # continue from here, just need to output images into directory OUTPUT and 
    # use imsave with cmap = 'gray' option to save as grayscale.


DATA = load_data("/home/satchel/Downloads/Project4-2/Project4/Data/Train")
compress_images(DATA, 1)