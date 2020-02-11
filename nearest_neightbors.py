###############################################################################
#
# AUTHOR(S): Samantha Muellner
# DESCRIPTION: program that will find and graph nearest_neighbors on the
#       provided data set -- in this case spam.data
# VERSION: 1.0.0v
#
###############################################################################

import numpy as np
import csv
from math import sqrt
from sklearn.neighbors import NearestNeighbors

# Function: kFoldCV
# INPUT ARGS:
#   x_mat : a matrix of numeric inputs (one row for each observation, one column for each feature)
#   y_vec : a vector of binary outputs (the corresponding label for each observation, either 0 or 1)
#   compute_prediction : a function that takes three inputs (X_train,y_train,X_new), 
#           trains a model using X_train,y_train, then outputs a vector of predictions 
#           (one element for every row of X_new).
#   fold_vector : a vector of integer fold ID numbers (from 1 to K).
# Return: error_vec
def kFoldCV(x_mat, y_vec, computePrediction, fold_vector):
    
    # numeric vector of size k
    error_vec = np.zeros(len(fold_vec))

    # loop over the unique values k in fold_vec
    for vector in fold_vector:
        # define X_new, y_new based on the observations for which the corr. elements
        #       of fold_vec are equal to the current fold ID k
        X_new = vector[0]
        y_new = vector[1]

        # define X_train, y_train using all the other observations
        X_train = 0
        y_train = 0

        # call ComputePredictions and store the result in a variable named pred_new
        pred_new = compute_predictions()

        # compute the zero-one loss of pred_new with respect to y_new
        #       and store the mean (error rate) in the corresponding entry error_vec


    return error_vec


# Function: NearestNeighborsCV
# INPUT ARGS:
#    X_mat : a matrix of numeric inputs/features (one row for each observation, one column for each feature).
#    y_vec : a vector of binary outputs (the corresponding label for each observation, either 0 or 1).
#    X_new : a matrix of numeric inputs/features for which we want to compute predictions.
#    num_folds : default value 5.
#    max_neighbors : default value 20
# Return: error_vec
def NearestNeighborsCV(X_Mat, y_vec, X_new, num_folds, max_neighbors):
    validation_fold_vec = np.array([num for num in range(1, num_folds + 1)])
    
    total_entries = num_folds * max_neighbors

    # numeric matrix (num_folds x max_neighbors)
    error_mat = weight_matrix = np.array(np
                        .zeros(total_entries)
                        .reshape(num_folds, max_neighbors))
    
    error_mat_index = 0

    # for loop over num_neighbors
    for index in max_neighbors:
        #call KFoldCV
        # specify ComputePredictions = function that uses your programs language's         
        error_mat[error_mat_index] = kFoldCV(X_Mat, y_vec, 
                            NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X), 
                            validation_fold_vec)
        error_mat_index += 1

    # will contain the mean of each column of error_matrix
    mean_error_vec = np.zeros(max_neighbors)
    column_count = 0

    # take mean of each column of error_mat and add to mean_error_vec
    for column in error_mat:
        mean_error_vec[column_count] = mean(column)
        column_count += 1

    # create variable that contains the number of neighbors with minimal error
    best_neighbors = 0

    # output:
    #   (1) the predictions for X_New, using the entire X_mat, y_vec with best_neighbors
    #   (2) the mean_error_mat for visualizing the validation error
        
