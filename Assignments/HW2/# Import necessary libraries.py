# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import glob
from PIL import Image

# Import the image data
# Define the path of the folder contains the images
images_folder = "/Users/thanapoomphatthanaphan/Education/Master Degree/Courses/CS559_Machine-Learning-Fundamentals-Applications/Assignments/HW2/face_data/*.bmp"

# Get a list, containing the path of each image
image_path = glob.glob(images_folder)

# Load the image objects, and put them in the list
images_data = []
for image in image_path:
    images_data.append(np.array(Image.open(image)))

# Perform PCA
def perform_pca(images_data, K):
    
    """A function to perform PCA"""
    
    # Reshape each image from 2D to 1D vector
    images_1d = []
    for image in images_data:
        images_1d.append(image.flatten())
    images_1d = np.array(images_1d)
    
    # Compute the mean image
    n_train = len(images_1d)
    mean_image = np.sum(images_1d) / n_train
    
    # Recenter each training image by subtracting with the mean image
    images_centered = images_1d - mean_image
    
    # Compute the covariance matrix
    cov_matrix = np.cov(images_centered, rowvar=False)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Order the eigenvalues and their corresponding eigenvectors in desceding order
    ordered_indices = np.argsort(eigenvalues)[::-1]
    ordered_eigenvalues = eigenvalues[ordered_indices]
    ordered_eigenvectors = eigenvectors[:, ordered_indices]
    
    # Select the top-K eigenfaces
    top_K_eigenfaces = ordered_eigenvectors[:, :K]
    
    # Visualize the top 10 eigenfaces
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(top_eigenfaces[:, i].reshape(256, 256))
        plt.title(f'Eigenface {i + 1}')
        plt.axis('off')
    
    plt.show()

# Call the function to perform PCA and visualize eigenfaces
K = 30
perform_pca(images_data[:157], K)