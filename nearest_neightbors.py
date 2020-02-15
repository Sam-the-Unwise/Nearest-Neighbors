###############################################################################
#
# AUTHOR(S): Samantha Muellner
# DESCRIPTION: program that will find and graph nearest_neighbors on the
#       provided data set -- in this case spam.data
# VERSION: 1.0.3v
#
###############################################################################

import numpy as np
import csv
from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors, datasets

# Function: kFoldCV
# INPUT ARGS:
#   x_mat : a matrix of numeric inputs (one row for each observation, one column for each feature)
#   y_vec : a vector of binary outputs (the corresponding label for each observation, either 0 or 1)
#   compute_prediction : a function that takes three inputs (X_train,y_train,X_new), 
#           trains a model using X_train,y_train, then outputs a vector of predictions 
#           (one element for every row of X_new).
#   fold_vector : a vector of integer fold ID numbers (from 1 to K).
# Return: error_vec
def kFoldCV(x_mat, y_vec, compute_prediction, fold_vector):
    
    # numeric vector of size k
    error_vec = np.zeros(len(fold_vector))
    index_count = 0

    # loop over the unique values k in fold_vec
    for vector in fold_vector:
        # define X_new, y_new based on the observations for which the corr. elements
        #       of fold_vec are equal to the current fold ID k
        X_new = x_mat[vector]
        y_new = vector

        # define X_train, y_train using all the other observations
        X_train = 0
        y_train = 0

        # call ComputePredictions and store the result in a variable named pred_new
        pred_new = compute_prediction(X_train, y_train, X_new)

        # compute the zero-one loss of pred_new with respect to y_new
        #       and store the mean (error rate) in the corresponding entry error_vec
        zero_one_vector = pred_new.kneighbors_graph(x_mat).toarray()

        #the below code doesn't work because of the way that zero_one_vector is stored
        error_vec[index_count] = zero_one_vector
        
        index_count += 1

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
    # array with numbers 1 to k 
    #       k = length of num_folds
    validation_fold_vec = np.array([num for num in range(1, num_folds + 1)])
    
    total_entries = num_folds * max_neighbors

    # numeric matrix (num_folds x max_neighbors)
    error_mat = weight_matrix = np.array(np
                        .zeros(total_entries)
                        .reshape(num_folds, max_neighbors))
    
    error_mat_index = 0

    for index in range(1, max_neighbors):
        # call KFoldCV, and specify ComputePreditions = a function that uses 
        #    your programming language’s default implementation of the nearest 
        #    neighbors algorithm, with num_neighbors 

        # store error rate vector in error_mat
        error_mat[error_mat_index] = kFoldCV(X_Mat, y_vec, 
                            NearestNeighbors, 
                            validation_fold_vec)

        error_mat_index += 1

    # will contain the mean of each column of error_matrix
    mean_error_vec = np.zeros(max_neighbors)
    column_count = 0

    # take mean of each column of error_mat and add to mean_error_vec
    for column in error_mat:
        mean_error_vec[column_count] = np.mean(column)
        column_count += 1

    # create variable that contains the number of neighbors with minimal error
    best_neighbors = 0

    # output:
    #   (1) the predictions for X_New, using the entire X_mat, y_vec with best_neighbors
    #   (2) the mean_error_mat for visualizing the validation error
    
    return X_new, error_mat

# Function: convert_data_to_matrix
# INPUT ARGS:
#   file_name : the csv file that we will be pulling our matrix data from
# Return: data_matrix_full
def convert_data_to_matrix(file_name):
    with open(file_name, 'r') as data_file:
        spam_file = list(csv.reader(data_file, delimiter = " "))

    data_matrix_full = np.array(spam_file[0:], dtype=np.float)
    return data_matrix_full


# Function: main
# INPUT ARGS:
#   [none]
# Return: [none]
def main():
    # get the data from our CSV file
    data_matrix_full = convert_data_to_matrix("spam.data")

    np.random.shuffle(data_matrix_full)

    # get necessary variables
    # shape yields tuple : (row, col)
    col_length = data_matrix_full.shape[1]
    
    X_Mat = np.delete(data_matrix_full, col_length - 1, 1)
    y_vec = data_matrix_full[:,57]

    X_new = np.array([])

    # print X_Mat row 1 column 0
    #print(X_Mat[1][0])

    X_new_predictions, mean_error_mat = NearestNeighborsCV(X_Mat, y_vec, X_new, 5, 20)

    # TO DO:

    # plot the validation error as a function of the number of neighbors
    #       separately for each fold
    #   - Draw a bold line for the mean validation error 
    #   - draw a point to emphasize the minimum

    # Randomly createa  variable test_vold_vec
    #   - a vector with one element for each observations
    #       - elements are integers from 1 to 4
    # include a table of counts with a row for each fold (1/2/3/4) and a column 
    #   for each class (0/1)

    # Use KFoldCV with three algorithms: 
    #       (1) baseline/underfit – predict most frequent class, 
    #       (2) NearestNeighborsCV, 
    #       (3) overfit 1-nearest neighbors model. Plot the resulting test 
    #               error values as a function of the data set, in order to 
    #               show that the NearestNeighborsCV is more accurate than 
    #               the other two models. Example:

main()