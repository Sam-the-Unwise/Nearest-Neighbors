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
import random

NUM_FOLDS = 4
MAX_NEIGHBORS = 20

PRED_NEW_INDEX = 0
Y_NEW_INDEX = 1

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
def kFoldCV(x_mat, y_vec, compute_prediction, fold_vector, current_n_neighbors, pred_dict):

    # numeric vector of size k
    error_vec = []
    col = x_mat.shape[1]
    row = x_mat.shape[0]
    eighty_percent_row = int(row * .8)
    twenty_percent_row = int(row * .2)


    y_new_info = {}
    y_pred_info = {}

    array_of_zeros = [0] * (col)



    # loop over the unique values k in fold_vec
    for foldIDk in range(1, NUM_FOLDS + 1):
        # define X_train, y_train using all the other observations
        X_train = np.array(array_of_zeros).reshape(1, col)
        y_train = np.zeros(1)

        # define X_new, y_new to contain the corr. folds of foldIDk to num_folds
        X_new = np.array(array_of_zeros).reshape(1, col)
        y_new = np.zeros(1)

        one_items_in_x_new = 0


        new_index = 0
        train_index = 0

        # define X_new, y_new based on the observations for which the corr.
        #       elements of fold_vec are equal to the current fold ID k
        index = 0
        for num in fold_vector:

            if num == foldIDk:
                # # account for possibility that our row amount doesn't divide evenly into 80% and 20%
                # # ex: 4601 yields 80%: 3680 and 20%:920, meaning we didn't account for the last 1

                y_new = np.append(y_new, y_vec[index])
                X_new = np.vstack((X_new, x_mat[index,:]))

                new_index += 1

            else:

                y_train = np.append(y_train, y_vec[index])
                X_train = np.vstack((X_train, x_mat[index,:]))

                train_index += 1

            index += 1

        # delete zero elements used to initialize the arrays
        X_new = np.delete(X_new, 1, 0)
        X_train = np.delete(X_train, 1, 0)
        y_train = np.delete(y_train, 1, 0)
        y_new = np.delete(y_new, 1, 0)


        # call ComputePredictions and store the result in a variable named
        #       pred_new
        pred_new = compute_prediction.fit(X_train, y_train).predict(X_new)

        y_new_info[foldIDk] = y_new
        y_pred_info[foldIDk] = pred_new

        # compute the zero-one loss of pred_new with respect to y_new
        #       and store the mean (error rate) in the corresponding entry
        #       error_vec
        error_vec.append(zero_one_loss(y_new, pred_new))

    pred_dict[current_n_neighbors] = [y_pred_info, y_new_info]

    return error_vec, pred_dict


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

    extra_folds_to_add = 0

    validation_fold_vec = np.array(list(np.arange(1,
                                        num_folds + 1))
                                        * multiplier_of_num_folds)

    # make sure that validation_fold_vec is the same size as X_Mat.shape[0]
    while validation_fold_vec.shape[0] != X_Mat.shape[0]:
        validation_fold_vec = np.append(validation_fold_vec, random.randint(1, num_folds))

    np.random.shuffle(validation_fold_vec)

    # numeric matrix (num_folds x max_neighbors)
    error_mat = {}
    error_mat_index = 0
    current_n_neighbors = 1

    # create a dictionary to help gather information about the predicted sets
    # format this is saved in is:
    #   {num_neighbors :
    #       [
    #           {foldIDk: [pred_new], foldIDk: [pred_new]},
    #           {foldIDk: [y_new], foldIDk: [y_new]}
    #       ]
    #   }
    predictions_for_x_new = {}

    for num_neighbors in range(1, max_neighbors + 1):
        # call KFoldCV, and specify ComputePreditions = a function that uses
        #    your programming language’s default implementation of the nearest
        #    neighbors algorithm, with num_neighbors

        # store error rate vector in error_mat
        error_mat[num_neighbors], predictions_for_x_new = kFoldCV(X_Mat,
                                        y_vec,
                                        KNeighborsClassifier(n_neighbors = 1),
                                        validation_fold_vec,
                                        current_n_neighbors,
                                        predictions_for_x_new)

        # variable for CSV formating
        current_n_neighbors += 1

        error_mat_index += 1

    # save error vector to a csv
    with open("1NN_percent_error.csv", mode = 'w') as roc_file:

        fieldnames = ['num neighbors', 'fold', 'error']
        writer = csv.DictWriter(roc_file, fieldnames = fieldnames)

        writer.writeheader()

        # get the fold number
        for neighbor_number, error_list in error_mat.items():
            for foldID in range(1, num_folds + 1):
                error_val = error_list[foldID - 1]
                writer.writerow({'num neighbors': neighbor_number,
                                'fold': foldID,
                                'error': error_val})


    # will contain the mean of each column of error_matrix
    mean_error_vec = {}
    column_count = 0

    # take mean of each column of error_mat and add to mean_error_vec
    for number, column in error_mat.items():
        mean_error_vec[number] = np.mean(column)*100

    # create variable that contains the number of neighbors with minimal error
    best_neighbor = 1
    best_percentage = mean_error_vec[1]

    worst_neighbor = 1
    worst_percentage = mean_error_vec[1]

    # get the number of neighbors that has the minimal amount of error
    for item, value in mean_error_vec.items():

        if value < best_percentage:
            best_neighbor = item
            best_percentage = value

        if value > worst_percentage:
            worst_neighbor = item
            worst_percentage = value

    print("The best neighbor is " + str(best_neighbor)
            + " because it had the least error rate at " + str(best_percentage) + "\n")

    # output:
    #   (1) the predictions for X_New, using the entire X_mat, y_vec with
    #       best_neighbors
    #   (2) the mean_error_mat for visualizing the validation error
    return [predictions_for_x_new, mean_error_vec]



# Function: convert_data_to_matrix
# INPUT ARGS:
#   file_name : the csv file that we will be pulling our matrix data from
# Return: data_matrix_full
def convert_data_to_matrix(file_name):
    with open(file_name, 'r') as data_file:
        spam_file = list(csv.reader(data_file, delimiter = " "))

    data_matrix_full = np.array(spam_file[0:], dtype=np.float)
    return data_matrix_full

# Function: baseline_pred
# INPUT ARGS:
#   y_vec: vector of spam or not
# Return: vector of most common prediction
def baseline_pred(y_vec):
    baseline_pred_vec = np.zeros(len(y_vec)).reshape(len(y_vec),1)
    unique_num, counts = np.unique(y_vec, return_counts=True)
    if(counts[1] > counts[0]):
        return baseline_pred_vec.ones(len(y_vec)).reshape(len(y_vec),1)
    return baseline_pred_vec



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

    max_neighbors = MAX_NEIGHBORS

    list_of_elements = NearestNeighborsCV(X_sc,
                                            y_vec,
                                            X_new,
                                            NUM_FOLDS,
                                            max_neighbors)



    ############################## TEST FOLD VEC ##############################

    # necessary variables to create test_fold_vec
    num_folds = 4
    multiplier_of_num_folds = int(X_Mat.shape[0]/num_folds)

    test_fold_vec = np.array(list(np.arange(1,
                                        num_folds + 1))
                                        * multiplier_of_num_folds)

    # make sure that test_fold_vec is the same size as X_Mat.shape[0]
    while test_fold_vec.shape[0] != X_sc.shape[0]:
        test_fold_vec = np.append(test_fold_vec, random.randint(1, num_folds))

    np.random.shuffle(test_fold_vec)

    # create table that will display how many 0s and 1s are in each fold
    dict_of_folds_to_classifiers = {}

    # initialize dict_of_folds values to 0
    for num in range(1, num_folds + 1):
        dict_of_folds_to_classifiers[num] = [0, 0]

    # get count of how many 0s and 1s are in each fold case
    # store in dictionary in format {fold_num: [zero_count, ones_count], ....}
    fold_num_index = 0
    for fold_num in test_fold_vec:

        if y_vec[fold_num_index] == 0:
            dict_of_folds_to_classifiers[fold_num][0] += 1

        elif y_vec[fold_num_index] == 1:
            dict_of_folds_to_classifiers[fold_num][1] += 1

        fold_num_index += 1


    # print out table
    print("Num Folds vs Classes")
    print("------------------------------")
    print("{: <10s} | {: <10s} | {: <10s}".format("fold num", "0", "1"))

    for fold, count_list in dict_of_folds_to_classifiers.items():
        print("{: <10s} | {: <10s} | {: <10s}".format(str(fold), str(count_list[0]), str(count_list[1])))




    ############################ SAVE INFO TO CSVS ############################

    # get elements needed from our list_of_elements
    X_new_predictions = list_of_elements[0]
    mean_error_mat = list_of_elements[1]


    ######################## SAVE PREDICTION DICTIONARY #######################

    # save X_new_predictions to a csv
    #   this csv will contain the number of neighbors
    #        corresponding to the number of folds
    #        corresponding to the predicted y val
    #        corresponding to the actual y value
    #        corresponding to if it was an accurate prediction
    with open("1NN_prediction_dictionary.csv", mode = 'w') as roc_file:

        fieldnames = ['num neighbors', 'num folds', 'predicted y',
                        'actual y', 'accurate prediction']
        writer = csv.DictWriter(roc_file, fieldnames = fieldnames)

        writer.writeheader()

        # loop through the datasets for all the neighbors
        for num_of_neighbors in range(1, max_neighbors + 1):

            # get the list corresponding to specific neighbor
            neighbor_values = X_new_predictions[num_of_neighbors]

            # get dictionaries from list
            prediction_vector = neighbor_values[PRED_NEW_INDEX]
            actual_y_vector = neighbor_values[Y_NEW_INDEX]

            num_fold = 1

            # loop through the number of folds for each neighbor amount
            for num_fold in range(1, NUM_FOLDS + 1):
                prediction_values = prediction_vector[num_fold]
                actual_y_values = actual_y_vector[num_fold]

                # loop through an index for each of the values in each ..values vector
                for index in range(len(prediction_values)):
                    pred_val = prediction_values[index]
                    actual_val = actual_y_values[index]

                    accurate_prediction = pred_val == actual_val

                    writer.writerow({'num neighbors' : num_of_neighbors,
                                    'num folds' : num_fold,
                                    'predicted y' : pred_val,
                                    'actual y' : actual_val,
                                    'accurate prediction' : accurate_prediction})



    ########################## CREATE MEAN ERROR CSV ##########################
    with open("1NN_mean_error.csv", mode = 'w') as roc_file:

        fieldnames = ['num neighbors', 'mean error']
        writer = csv.DictWriter(roc_file, fieldnames = fieldnames)

        writer.writeheader()

        for neighbor, error in mean_error_mat.items():
            writer.writerow({'num neighbors': neighbor,
                            'mean error': error})


    # write information for the X_new_predictions and x_error_mat to a csv file

    # TO DO:

    # plot the validation error as a function of the number of neighbors
    #       separately for each fold
    #   - Draw a bold line for the mean validation error
    #   - draw a point to emphasize the minimum


    # Use KFoldCV with three algorithms:
    #       (1) baseline/underfit – predict most frequent class,
    #       (2) NearestNeighborsCV (done),
    #       (3) overfit 1-nearest neighbors model.
    #
    #
    # Plot the resulting test error values as a function of the data set, in order to
    #       show that the NearestNeighborsCV is more accurate than the other two models

main()