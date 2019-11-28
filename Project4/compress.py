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

    Z = pca.compute_Z(DATA)
    cov_z = pca.compute_covariance_matrix(Z)
    L, PCS = pca.find_pcs(cov_z)
    Z_star = pca.project_data(Z, PCS, L, k, 0)
    # Get the first k principal components:

    U_T = PCS.transpose()[:k]
    # At this point, U_TS is of size 2880 x k since there are 2880 'features' (our dimension)
    compressed_images = np.dot(Z_star, U_T)
    # continue from here, just need to output images into directory OUTPUT and 
    # use imsave with cmap = 'gray' option to save as grayscale.
    for x in range(len(DATA)):
        output = "/home/satchelh/Desktop/CS491/test_output/image" + str(x) + ".png"
        plt.imsave(output, compressed_images[x].reshape(60, 48), cmap='gray')

DATA = load_data("/home/satchelh/Desktop/CS491/Project4-2/Project4/Data/Train")
compress_images(DATA, 100)