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
    Calculates the cluster center for a given cluster X. Rounds to 3 decimal points.
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
        mean[i] = round(mean[i] / len(X), 3)

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


def compute_random_cluster_center(minima, maxima):
    """
    Compute a random cluster center within the range of the data. Round to 3 decimal points.
    :param minima:
    :param maxima:
    :return: cluster center
    """
    cluster_center = []

    for i in range(len(minima)):
        cluster_center.append(round(random.uniform(minima[i], maxima[i]), 3))

    return cluster_center


def find_extrema(X):
    minima = []
    maxima = []

    for point_index in range(len(X)):
        for feature_index in range(len(X[0])):
            if point_index == 0:
                minima.append(X[point_index][feature_index])
                maxima.append(X[point_index][feature_index])
            else:
                if X[point_index][feature_index] < minima[feature_index]:
                    minima[feature_index] = X[point_index][feature_index]

                if X[point_index][feature_index] > maxima[feature_index]:
                    maxima[feature_index] = X[point_index][feature_index]

    return minima, maxima


def K_Means(X, K):
    points = X.tolist()
    minima, maxima = find_extrema(X)

    # cluster_center_assignments: list with each (index-1) corresponding to a cluster center k. At each index is a
    # list with corresponding points
    cluster_center_assignments = []
    for i in range(K):
        cluster_center_assignments.append([])

    # list of cluster centers
    # randomly initialize cluster centers: choose random points from the data set X
    cluster_centers = random.choices(points, k=K)

    # emulating a do while loop
    cluster_centers_have_changed = False
    while True:
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
                new_cluster_center = compute_random_cluster_center(minima, maxima)

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
