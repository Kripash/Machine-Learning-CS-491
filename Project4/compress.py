import os
import matplotlib.pyplot as plt
import numpy as np
import pca
import copy

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
    mean = np.mean(DATA, axis=0)
    #print(mean)
    Z = pca.compute_Z(DATA)
    cov_z = pca.compute_covariance_matrix(Z)
    L, PCS = pca.find_pcs(cov_z)
    Z_star = pca.project_data(Z, PCS, L, k, 0)
    # Get the first k principal components:

    U_T = PCS.transpose()[:k]
    # At this point, U_TS is of size 2880 x k since there are 2880 'features' (our dimension)
    compressed_images = np.dot(Z_star, U_T)
    #If we return additional things from pca.compute_Z, we can assign them to variables that are initialized to none
    #And do a check to see if they are not None: e.g
    """
    mean = None 
    scaling = None
    Z, mean, scaling = pca.compute_Z(DATA)
    if(mean is not None):
        compressed_images = compressed_images + mean
    if(scaling is not None):
        compressed_images = compressed_images * scaling
    """
    compressed_images = compressed_images + mean
    # continue from here, just need to output images into directory OUTPUT and 
    # use imsave with cmap = 'gray' option to save as grayscale.
    for x in range(len(DATA)):
        output = "C:/Tutology/Machine-Learning-CS-491-/Project4/Project4/Data/test_output/image" + str(x) + ".png"
        plt.imsave(output, compressed_images[x].reshape(60, 48), cmap='gray')

DATA = load_data("C:\Tutology\Machine-Learning-CS-491-\Project4\Project4\Data\Train")
compress_images(DATA, 100)