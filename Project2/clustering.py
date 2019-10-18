import numpy as np
import math
import random


def calculate_euclidean_distance(x, cluster_center):
    """
    :param x: a specific data point (python list)
    :param cluster_center: a specific cluster center (python list)
    """
    squared_sum = 0

    for i in range(len(x)):
        squared_sum += (cluster_center[i] - x[i])**2

    return math.sqrt(squared_sum)


def find_nearest_cluster_center(x, cluster_centers):
    """
    :param x: a data point
    :param cluster_centers: list of all cluster centers
    :return: index of the cluster center in cluster_centers that has the minimum distance to x
    """
    min_distance = math.inf
    cluster_center_index = 0

    for i in range(len(cluster_centers)):
        distance = calculate_euclidean_distance(x, cluster_centers[i])
        if distance <= min_distance:
            min_distance = distance
            cluster_center_index = i

    return cluster_center_index


def calculate_mean_of_points(X):
    """
    Calculates the cluster center for a given cluster X. Rounds to 2 decimal points.
    :param X: list of points in a cluster.
    :return: mean. cluster center for the given cluster X
    """
    mean = []

    for point_index in range(len(X)):
        for feature_index in range(len(X[0])):
            if point_index == 0:
                mean.append(X[point_index][feature_index])
            else:
                mean[feature_index] += X[point_index][feature_index]

    for i in range(len(mean)):
        mean[i] = round(mean[i] / len(X), 2)

    return mean


def find_points_in_cluster(X, cluster_center_index, cluster_center_assignments):
    """
    :param X:
    :param cluster_center_index:
    :param cluster_center_assignments:
    :return: all points that belong to a cluster with cluster_center_index
    """
    points_in_cluster = []

    for i in range(len(cluster_center_assignments[cluster_center_index])):
        points_in_cluster.append(X[cluster_center_assignments[cluster_center_index][i]])

    return points_in_cluster


def K_Means(X, K):
    points = X.tolist()

    # cluster_center_assignments: list with each (index-1) corresponding to a cluster center k. At each index is a
    # list with corresponding points
    cluster_center_assignments = []

    # list of cluster centers
    # randomly initialize cluster centers: choose random points from the data set X
    cluster_centers = random.choices(points, k=K)

    # emulating a do while loop
    cluster_centers_have_changed = False
    while True:
        # reset the cluster_center_assignments since the will be recaluclated
        cluster_center_assignments = []
        for i in range(K):
            cluster_center_assignments.append([])

        # for points[i] in points
        for i in range(len(points)):
            # compute euclidean distance from data point X[i] to every cluster center
            nearest_cluster_center_index = find_nearest_cluster_center(points[i], cluster_centers)
            # assign points[i] to closest cluster center
            cluster_center_assignments[nearest_cluster_center_index].append(i)

        # for every cluster center
        for i in range(len(cluster_centers)):
            # compute the mean of all points assigned to that cluster center => new cluster center
            points_in_cluster = find_points_in_cluster(X, i, cluster_center_assignments)
            new_cluster_center = calculate_mean_of_points(points_in_cluster)

            # if new_cluster_center is empty, assign a random cluster center within the range of the data
            # this can happen if a cluster becomes empty
            if not new_cluster_center:
                new_cluster_center = random.choice(points)

            if new_cluster_center != cluster_centers[i]:
                cluster_centers_have_changed = True
                cluster_centers[i] = new_cluster_center

        # emulating a do while loop: while cluster centers stop changing
        if not cluster_centers_have_changed:
            break
        else:
            cluster_centers_have_changed = False

    # return cluster_centers as numpy array
    return np.array(cluster_centers)


def find_majority(all_cluster_centers, amounts, index):
    """
    Find and return the cluster centers that came up the most.
    :param all_cluster_centers:
    :param amounts:
    :param index:
    :return: majority_cluster_centers
    """
    majority_cluster_centers = []
    number_of_kmeans_computations = index
    max_occurrences = 0
    number_of_results_for_max_occurrences = 0  # There are no majority cluster_centers if there are multiple different solutions

    for i in range(len(all_cluster_centers)):
        if i == 0:
            majority_cluster_centers = all_cluster_centers[i]
            max_occurrences = amounts[i]
            number_of_results_for_max_occurrences = 1
        else:
            if amounts[i] > max_occurrences:
                majority_cluster_centers = all_cluster_centers[i]
                max_occurrences = amounts[i]
                number_of_results_for_max_occurrences = 1
            elif amounts[i] == max_occurrences:
                majority_cluster_centers = []  # There is no majority at this moment
                number_of_results_for_max_occurrences += 1

    return majority_cluster_centers


def K_Means_better(X,K):
    majority_cluster_centers = []
    all_cluster_centers = []
    amount_of_each_of_all_cluster_centers = []

    there_is_a_majority = False
    index = 1
    max_iterations = 5000  # we don't want to wait for hours...
    min_iterations = 1000  # but we should still try kmeans many times
    while True:
        cluster_centers = K_Means(X, K).tolist()

        # Order cluster_centers by first value (x0), then x1, then x2, ... in order to be serving as a distinct key in
        # the list all_cluster_centers for the list amount_of_each_of_all_cluster_centers
        cluster_centers = sorted(cluster_centers)

        # Save the number of times a cluster center is the result of K_Means(X,Y)
        if cluster_centers not in all_cluster_centers:
            all_cluster_centers.append(cluster_centers)
            amount_of_each_of_all_cluster_centers.append(1)
        else:
            amount_of_each_of_all_cluster_centers[all_cluster_centers.index(cluster_centers)] += 1

        # find out if there is a majority
        if index >= min_iterations:
            majority_cluster_centers = find_majority(all_cluster_centers, amount_of_each_of_all_cluster_centers, index)

        if majority_cluster_centers is not None and majority_cluster_centers:
            there_is_a_majority = True
        else:
            there_is_a_majority = False
            majority_cluster_centers = []

        if (there_is_a_majority and index >= min_iterations) or index == max_iterations:
            break

        index += 1

    # return majority_cluster_centers as numpy array
    return np.array(majority_cluster_centers)
