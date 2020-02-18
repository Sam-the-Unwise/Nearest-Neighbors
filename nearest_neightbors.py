###############################################################################
#
# AUTHOR(S): Samantha Muellner
# DESCRIPTION: program that will find and graph nearest_neighbors on the
#       provided data set -- in this case spam.data
# VERSION: 1.0.3v
#
###############################################################################

import numpy as np
import csv, math
from math import sqrt
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import zero_one_loss
from sklearn import neighbors, datasets

NUM_FOLDS = 5

TRUE = 1
FALSE = 0

# Function: kFoldCV
# INPUT ARGS:
#   x_mat : a matrix of numeric inputs (one row for each observation, one column for each feature)
#   y_vec : a vector of binary outputs (the corresponding label for each observation, either 0 or 1)
#   compute_prediction : a function that takes three inputs (X_train,y_train,X_new),
#           trains a model using X_train,y_train, then outputs a vector of predictions
#           (one element for every row of X_new).
#   fold_vector : a vector of integer fold ID numbers (from 1 to K).
# Return: error_vec
def kFoldCV(x_mat, y_vec, compute_prediction, fold_vector, current_n_neighbors):

    # numeric vector of size k
    error_vec = []
    col = x_mat.shape[1]
    row = x_mat.shape[0]
    eighty_percent_row = int(row*.8)
    twenty_percent_row = int(row*.2)

    while eighty_percent_row + twenty_percent_row != row:
        eighty_percent_row += 1


    # print header for printing out "table" containing prediction vector and actual vector
    #print("pred_y  |  real_y  |  is_error")
    print('{: <10s}| {: <10s}| {: <10s}| {: <10s}| {}'.format("fold ID",
                                                                "num neighbors",
                                                                "pred_y",
                                                                "real_y",
                                                                "is_error"))

    # loop over the unique values k in fold_vec
    for foldIDk in range(1, NUM_FOLDS + 1):
        # define X_train, y_train using all the other observations
        X_train = np.zeros(col*eighty_percent_row).reshape(eighty_percent_row,
                                                        col)
        y_train = np.zeros(eighty_percent_row)

        # define X_new, y_new to contain the corr. folds of foldIDk to num_folds
        X_new = np.zeros(col*twenty_percent_row).reshape(twenty_percent_row, 
                                                        col)
        y_new = np.zeros(twenty_percent_row)

        one_items_in_x_new = 0


        new_index = 0
        train_index = 0

        # define X_new, y_new based on the observations for which the corr.
        #       elements of fold_vec are equal to the current fold ID k
        index = 0
        for num in fold_vector:

            if num == foldIDk:
                y_new[new_index] = y_vec[index]
                X_new[new_index] = x_mat[index,:]

                new_index += 1

            else:
                y_train[train_index] = y_vec[index]
                X_train[train_index] = x_mat[index,:]

                train_index += 1

            index += 1

        # call ComputePredictions and store the result in a variable named
        #       pred_new
        pred_new = compute_prediction.fit(X_train, y_train).predict(X_new)


        # print out the actual vector and the prediction vector
        for index in range(len(pred_new)):
            is_error = pred_new[index] != y_new[index]
            print('{: <10s}| {: <10s}| {: <10s}| {: <10s}| {}'
                                                    .format(str(foldIDk),
                                                    str(current_n_neighbors),
                                                    str(pred_new[index]),
                                                    str(y_new[index]),
                                                    is_error))

        # compute the zero-one loss of pred_new with respect to y_new
        #       and store the mean (error rate) in the corresponding entry
        #       error_vec
        # use 100 - loss*100 in order to get percent instead of loss
        error_vec.append(100 - (zero_one_loss(y_new, pred_new) * 100))

    return error_vec


# Function: NearestNeighborsCV
# INPUT ARGS:
#    X_mat : a matrix of numeric inputs/features (one row for each observation,
#               one column for each feature).
#    y_vec : a vector of binary outputs (the corresponding label for each
#               observation, either 0 or 1).
#    X_new : a matrix of numeric inputs/features for which we want to compute
#               predictions.
#    num_folds : default value 5.
#    max_neighbors : default value 20
# Return: error_vec
def NearestNeighborsCV(X_Mat, y_vec, X_new, num_folds, max_neighbors):
    # array with numbers 1 to k
    #       k = length of num_folds
    multiplier_of_num_folds = int(X_Mat.shape[0]/num_folds)

    # extra_folds_to_add = 0
    # # make sure that validation_fold_vec is the same size as X_Mat.shape[0]
    # while num_folds * multiplier_of_num_folds != X_Mat.shape[0]:
    #     extra_folds_to_add += 1

    validation_fold_vec = np.array(list(np.arange(1,
                                        num_folds + 1))
                                        *multiplier_of_num_folds)
    np.random.shuffle(validation_fold_vec)

    # numeric matrix (num_folds x max_neighbors)
    error_mat = {}
    error_mat_index = 0
    current_n_neighbors = 1

    for num_neighbors in range(1, max_neighbors + 1):
        # call KFoldCV, and specify ComputePreditions = a function that uses
        #    your programming language’s default implementation of the nearest
        #    neighbors algorithm, with num_neighbors

        # store error rate vector in error_mat
        error_mat[num_neighbors] = kFoldCV(X_Mat,
                                        y_vec,
                                        KNeighborsClassifier(n_neighbors = num_neighbors),
                                        validation_fold_vec,
                                        current_n_neighbors)

        # variable for CSV formating
        current_n_neighbors += 1

        error_mat_index += 1

    # will contain the mean of each column of error_matrix
    mean_error_vec = {}
    column_count = 0

    # take mean of each column of error_mat and add to mean_error_vec
    for number, column in error_mat.items():
        mean_error_vec[number] = np.mean(column)

    # create variable that contains the number of neighbors with minimal error
    best_neighbor = 1
    best_percentage = mean_error_vec[1]

    # get the number of neighbors that has the minimal amount of error
    for item, value in mean_error_vec.items():
        if value > best_percentage:
            best_percentage = value
            best_neighbor = item

    # output:
    #   (1) the predictions for X_New, using the entire X_mat, y_vec with
    #       best_neighbors
    #   (2) the mean_error_mat for visualizing the validation error
    return X_new, best_neighbor, error_mat



# Function: convert_data_to_matrix
# INPUT ARGS:
#   file_name : the csv file that we will be pulling our matrix data from
# Return: data_matrix_full
def convert_data_to_matrix(file_name):
    with open(file_name, 'r') as data_file:
        spam_file = list(csv.reader(data_file, delimiter = " "))

    data_matrix_full = np.array(spam_file[0:4000], dtype=np.float)
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

    X_sc = scale(X_Mat)
    X_new = np.array([])

    max_neighbors = 20

    X_new_predictions, best_neighbor, mean_error_mat = NearestNeighborsCV(X_sc,
                                                            y_vec,
                                                            X_new,
                                                            NUM_FOLDS,
                                                            max_neighbors)

    # write information for the X_new_predictions and x_error_mat to a csv file

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