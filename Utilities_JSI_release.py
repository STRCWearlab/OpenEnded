from munkres import Munkres
import numpy as np
import re
import copy
import matplotlib.pyplot as plt
import math
import scipy
import six
from matplotlib import colors

color = list(six.iteritems(colors.cnames))

#bigger sigma closer to Euclidian
def kernel_gauss(x, y, sigma= 0.1):
    v = x - y
    l = math.sqrt(scipy.square(v).sum())
    return math.exp(-sigma * (l ** 2))


def kernel_normalise(k):
    return lambda x, y: k(x, y) / math.sqrt(k(x, x) + k(y, y))

def kernel_dist(x, y):
    # if gaussian kernel:
    return 2 - 2 * kernel(x, y)

kernel = kernel_normalise(kernel_gauss)

THRESHOLD = 16

def arrangeColors(colors_set):
    colors_set = [re.sub(r'\bwhite\b', 'nova', cs) for cs in colors_set]
    colors_set = [re.sub(r'\bindigo\b', 'aquamarine', cs) for cs in colors_set]
    colors_set = [re.sub(r'\bdarkseagreen\b', 'beige', cs) for cs in colors_set]
    colors_set = [re.sub(r'\bnova\b', 'indigo', cs) for cs in colors_set]
    return colors_set

#removes clusters which are contained by a bigger cluster
def removeContained(clusters):
    for i, cluster11 in enumerate(clusters):
        for j, cluster22 in enumerate(clusters):
            if (cluster11 == cluster22):
                continue
            if cluster11.timestampStart > cluster22.timestampStart and cluster11.timestampEnd < cluster22.timestampEnd:
                clusters.remove(cluster11)
    return clusters


def transform_activities(activity_list, THRESHOLD):
    i = 0
    counter = 0
    total_counter = 0
    segmented = []
    activity_order = []
    activity_array = []
    activities_coordinates = []
    for activity in activity_list:
        if (i==0):
            previous = activity
            i+=1
            continue
        if activity == previous:
            counter +=1
        else:
            if counter > THRESHOLD:#ToDo threshold
                segmented.append((total_counter, counter+1))
                activity_order.append(previous)
                activity_array.append([int(previous), [total_counter, total_counter + counter]])
            total_counter += counter+1
            counter = 0
        previous = activity
        i+=1
    segmented.append((total_counter, counter+1))
    activity_order.append(previous)
    activity_array.append([int(previous), [total_counter, total_counter + counter]])
    return [segmented, activity_order, activity_array]


def remove_small_activities(data, activity_list, THRESHOLD):
    i = 0
    counter = 0
    data_all = []
    data_segment = []
    for activity in activity_list:
        if (i==0):
            previous = activity
            data_segment.append(data[i])
            i += 1
            continue
        if activity == previous:
            counter +=1
            data_segment.append(data[i])
        else:
            if counter > THRESHOLD:#ToDo threshold
                data_all.extend(data_segment)
            data_segment = []
            data_segment.append(data[i])
            counter = 0
        previous = activity
        i+=1
    data_all.extend(data_segment)
    return np.asarray(data_all)


#Returns the mean values (centers) of each activity (if activity is repeated, then it is considered as the same activity)...
# ToDo: it should not be like this... each repetition is a separate activity
def get_activity_means(dataAll):
    keys = list(set(dataAll[:, [-1]].T[0]))
    activity_array_temp = []
    activity_dic = []
    for x in keys:
        activity_array_temp.append(dataAll[np.logical_or.reduce([dataAll[:, -1] == x])])
        activity_dic.append([x,dataAll[np.logical_or.reduce([dataAll[:, -1] == x])]])

    activity_means = []
    for item2 in activity_dic:
        item = item2[1]
        means = []
        for feature in item.T[:len(item.T)-2]:
            means.append(np.mean(feature))
        activity_means.append([item2[0], means])
    return activity_means


def calculate_accuracy(confusion_matrix_detailed, couter_data_samples_no_null):
    accuracy = 0
    recall = []
    precision = []
    sums = 0
    for i, row in enumerate(confusion_matrix_detailed[1:]):
        act_length = couter_data_samples_no_null[i]
        if act_length >= sum(row[:-1]):
            recall.append(row[i + 1] / float(act_length))
        else:
            recall.append(row[i + 1] / sum(row[:-1]))
        accuracy += row[i + 1]

    accuracy /= sum(couter_data_samples_no_null)

    for i, row in enumerate(confusion_matrix_detailed.T[1:-1]):
        if sum(row[1:]) == 0:
            precision.append(0.0)
        else:
            precision.append(float(row[i + 1]) / sum(row[1:]))

    if np.mean(precision) == 0 and np.mean(recall) == 0:
        f_measure = 0.0
    else:
        f_measure = 2 * np.mean(precision) * np.mean(recall) / (np.mean(precision) + np.mean(recall))

    return accuracy, recall, precision, f_measure


def calcualte_accuracy_hungarian(hungarian_matrix, couter_data_samples_no_null):
    hungarian_sum = 0
    cost_matrix = []
    recall = []
    precision = []
    for i, row in enumerate(hungarian_matrix):
        hungarian_sum += np.sum(row)
        cost_row = []
        for col in row:
            cost_row += [1000 - col]
        cost_matrix += [cost_row]
    m = Munkres()
    indexes = m.compute(cost_matrix)
    total = 0
    for row, column in indexes:
        value = hungarian_matrix[row][column]
        total += value
        recall.append(float(value)/couter_data_samples_no_null[row])
        if value == 0:
            precision.append(0)
        else:
            precision.append(float(value) / sum(hungarian_matrix[:, column]))
    accuracy = total / sum(couter_data_samples_no_null)

    while len(recall) < len (hungarian_matrix):
        recall.append(0.0)

    if np.mean(precision) == 0 and np.mean(recall) == 0:
        f_measure = 0.0
    else:
        f_measure = 2 * np.mean(precision) * np.mean(recall) / (np.mean(precision) + np.mean(recall))

    return accuracy, recall, precision, f_measure


def findClosestActivity(clusters,  activity_means, dict_activity_index_colour):
    cluster_segments = []
    cluster_segments_complex = []
    cluster_colors_set = []
    cluster_array = []
    ratios = []

    for cluster in clusters:
        #activity_means = np.asarray(activity_means)

        min_distance = kernel_dist(cluster.center, np.asarray(activity_means[0][1]))
        min_index = 0
        for mean2 in activity_means:
            mean =np.asarray( mean2[1])
            distance_temp = kernel_dist(cluster.center, mean)
            if min_distance > distance_temp:
                min_distance = distance_temp
                min_index = mean2[0]
        cluster_segments_complex.append(((int(cluster.timestampStart), (int(cluster.timestampEnd) - int(cluster.timestampStart))),color[dict_activity_index_colour[int(min_index)]][0]))
        cluster_array.append([dict_activity_index_colour[int(min_index)], [int(cluster.timestampStart), int(cluster.timestampEnd)]])

        ratios.append([dict_activity_index_colour[int(min_index)], int(cluster.timestampStart), int((cluster.num_points / cluster.size * 100) + 0.5) / 100.0])

    ratios  = sorted(ratios, key=lambda x: (-x[2], x[0]))
    #print "\nActivity, Movement_Ratio: \t" + str(ratios)
    cluster_segments_complex = sorted(cluster_segments_complex, key=lambda x: x[0])
    for cs in cluster_segments_complex:
        cluster_segments.append(cs[0])
        cluster_colors_set.append(cs[1])
    cluster_colors_set = arrangeColors(cluster_colors_set)

    return cluster_segments, cluster_segments_complex, cluster_colors_set, cluster_array, ratios


def findClosestClustersAndMerge(clusters):
    p = 1
    cluster_segments_complex2 = []
    cluster_segments2 = []
    cluster_colors_set2 = []
    skip = []
    merged = []
    min_cluster = []
    for iii, cluster1 in enumerate(clusters):
        if cluster1 in skip:
            continue
        min_distance = 1000
        min_index = index_temp = -1
        min_cluster = cluster1
        for cluster2 in clusters[iii + 1:]:
            if cluster2 in skip:
                continue
            distance_temp = kernel_dist(cluster1.center, cluster2.center)
            if min_distance > distance_temp:
                min_distance = distance_temp
                min_index = index_temp + iii + 1
                min_cluster = cluster2
            index_temp += 1
        if (abs(min_distance - cluster1.STD.mean) < p * cluster1.STD.std) or (
            abs(min_distance - min_cluster.STD.mean) < p * min_cluster.STD.std):
            # cluster_segments_complex2.append(((int(cluster1.timestampStart),
            #                                    (int(cluster1.timestampEnd) - int(cluster1.timestampStart))),
            #                                   color[int(iii)][0]))
            # cluster_segments_complex2.append(((int(min_cluster.timestampStart),
            #                                    (int(min_cluster.timestampEnd) - int(min_cluster.timestampStart))),
            #                                   color[int(iii)][0]))
            merged.append([(int(min_cluster.timestampStart),(int(min_cluster.timestampEnd) - int(min_cluster.timestampStart))),
                           (int(cluster1.timestampStart),(int(cluster1.timestampEnd) - int(cluster1.timestampStart)))])
            skip.append(min_cluster)
        else:
            if cluster1 in skip:
                continue
            # cluster_segments_complex2.append(((int(cluster1.timestampStart),
            #                                    (int(cluster1.timestampEnd) - int(cluster1.timestampStart))),
            #                                   color[int(iii)][0]))

    cluster_segments_complex2 = sorted(cluster_segments_complex2, key=lambda x: x[0])
    for cs in cluster_segments_complex2:
        cluster_segments2.append(cs[0])
        cluster_colors_set2.append(cs[1])
    cluster_colors_set2 = arrangeColors(cluster_colors_set2)
    cluster_colors_set2 = tuple(cluster_colors_set2)

    return merged, cluster_segments2, cluster_segments_complex2, cluster_colors_set2



def validation (cluster_colors_set, dataAll, dict_activity_index_colour, activities_set, cluster_segments_complex,
                ignore_cluster, null_label, cluster_array, cluster_intervals, n_clusters_, cluster_segments, threshold_cluster, VISUALIZATION):
    # validation, performance
    cluster_colors_set = tuple(cluster_colors_set)
    t = transform_activities(dataAll[:, [3]], threshold_cluster)
    activity_segments = t[0]
    activity_order = t[1]
    activity_array = t[2]

    colors_set = []
    for ind in activity_order:
        colors_set.append(color[int(dict_activity_index_colour[int(ind)])][0])
    colors_set = arrangeColors(colors_set)  # ToDo: find the null colour
    colors_set = tuple(colors_set)

    border = 0
    counter_activities = 0
    percentage_coverage = []
    percentage_coverage_max = []
    weights = []

    confusion_matrix = np.zeros((len(activities_set), len(activities_set) + 1))
    confusion_matrix_detailed = np.zeros((len(activities_set), len(activities_set) + 1))
    count_clusters_during_null = 0
    activities_not_null = []
    clusters_during_activities_no_null = []

    average_fragmentation_activities_same_color = []
    average_fragmentation_activities_diff_color = []
    used_clusters = []

    #FILTER Activity segments
    #ToDo
    for activity_item in activity_array:
        act_label = activity_item[0]
        act_interval = activity_item[1]

        if act_interval[1] == act_interval[0]: continue
        counter_activities += 1

    if ignore_cluster == True:
        hungarian_matrix = np.zeros((counter_activities, len(cluster_segments_complex)))
    else:
        hungarian_matrix = np.zeros((counter_activities, n_clusters_))

    counter_activities = 0
    for activity_item in activity_array:
        act_label = activity_item[0]
        act_interval = activity_item[1]

        if act_interval[1] == act_interval[0]: continue
        counter_activities += 1
        if not act_label in null_label:
            weights.append(act_interval[1] - act_interval[0])
            activities_not_null.append(activity_item)
        overlap_interval = 0
        overlap_interval_max = 0
        clusters_found_same_act_or_null = []
        cluster_found_complex = []
        clusters_during_activities_dif_color_no_null = []
        for h_i, cluster_item in enumerate(cluster_array):
            current_overlap_interval = 0
            clus_label = cluster_item[0]
            clus_interval = cluster_item[1]
            # if the cluster starts or ends inside the activity interval
            if (clus_interval[0] >= act_interval[0] - border and clus_interval[0] <= act_interval[1] + border) or (clus_interval[1] >= act_interval[0] - border and clus_interval[1] <= act_interval[1] + border):
                inttt = min(clus_interval[1], act_interval[1]) - max(clus_interval[0], act_interval[0])
                if inttt > threshold_cluster:
                    clusters_during_activities_dif_color_no_null.append(cluster_item)
                if ignore_cluster == True:
                    hungarian_matrix[counter_activities-1][h_i] += inttt if inttt > 0 else 0
                else:
                    hungarian_matrix[counter_activities-1][cluster_intervals[h_i][2]] += inttt if inttt > 0 else 0
                if clus_label == dict_activity_index_colour[act_label] or (clus_label in null_label):
                    clusters_found_same_act_or_null.append(clus_interval)
                    cluster_found_complex.append(cluster_item)
                    start = max(clus_interval[0], act_interval[0])
                    end = min(clus_interval[1], act_interval[1])
                    overlap_interval += float(end - start)
                    current_overlap_interval = float(end - start)
                    overlap_interval_max = overlap_interval
                else:
                    # if current_overlap_interval < (0.1 * act_interval[1] - act_interval[1]):
                    inttt = min(clus_interval[1], act_interval[1]) - max(clus_interval[0], act_interval[0])
                    confusion_matrix_detailed[dict_activity_index_colour[act_label]][clus_label] += inttt if inttt > 0 else 0
                    continue
            else:
                continue

        if overlap_interval < 0:
            overlap_interval = 0
            overlap_interval_max = 0

        # If multiple clusters are found
        if len(clusters_found_same_act_or_null) > 1:
            # find the biggest cluster inside the activity
            max_value = 0
            cluster_found_complex2 = copy.deepcopy(cluster_found_complex)
            for cl in cluster_found_complex2:
                cl[1][0] = max(cl[1][0], act_interval[0])
                cl[1][1] = min(cl[1][1], act_interval[1])

                # confusion_matrix_detailed[dict_activity_index_colour[act_label]][cl[0]] += cl[1][1] - cl[1][0]
                if cl[1][1] - cl[1][0] >= max_value:
                    max_value = cl[1][1] - cl[1][0]
                    max_act = cl[0]
            # fill the confusion matrix
            confusion_matrix[dict_activity_index_colour[act_label]][max_act] += 1
            overlap_interval_max = max_value

            # deal with overlapping clusters
            overlap_interval = 0
            cluster_found2 = copy.deepcopy(clusters_found_same_act_or_null)
            cluster_found2.sort()
            for cl in cluster_found2:
                cl[0] = max(cl[0], act_interval[0])
                cl[1] = min(cl[1], act_interval[1])
            cluster_found2 = sorted(cluster_found2)
            it = iter(cluster_found2)
            a, b = next(it)
            for c, d in it:
                if b >= c:
                    b = max(b, d)
                else:
                    overlap_interval += b - a
                    a, b = c, d
            overlap_interval += float(b - a)
            confusion_matrix_detailed[dict_activity_index_colour[act_label]][
                dict_activity_index_colour[act_label]] += overlap_interval
        # 1 or 0 clusters were found
        else:
            # if 1 cluster is found
            if len(cluster_found_complex) == 1:
                confusion_matrix[dict_activity_index_colour[act_label]][cluster_found_complex[0][0]] += 1
                confusion_matrix_detailed[dict_activity_index_colour[act_label]][
                    cluster_found_complex[0][0]] += overlap_interval
                # if (cluster_found_complex[0][0] in null_label) and not (act_label in null_label):
                #     print "Activity not found..." + str(dict_activity_index_colour[act_label]) + "\t" +str(activity_item) + "\tthe size of the activity is: " + str(act_interval[1] - act_interval[0])

        # check if there is a cluster bigger than this activity (bursts of the same activity covered by the same cluster)
        found = 0
        for h_ii, clusterrr in enumerate(cluster_array):
            if clusterrr in cluster_found_complex:
                continue
            clus_label = clusterrr[0]
            clus_interval = clusterrr[1]
            if act_interval[0] > clus_interval[0] and act_interval[1] < clus_interval[1]:
                confusion_matrix[dict_activity_index_colour[act_label]][clus_label] += 1
                confusion_matrix_detailed[dict_activity_index_colour[act_label]][clus_label] += act_interval[1] - act_interval[0]
                if ignore_cluster == True:
                    hungarian_matrix[counter_activities-1][h_ii] += act_interval[1] - act_interval[0]
                else:
                    hungarian_matrix[counter_activities-1][cluster_intervals[h_ii][2]] += act_interval[1] - act_interval[0]
                clusters_during_activities_dif_color_no_null.append(clusterrr)
                if clus_label == dict_activity_index_colour[act_label] or (act_label in null_label):
                    overlap_interval = act_interval[1] - act_interval[0]
                    overlap_interval_max = act_interval[1] - act_interval[0]
                    found = 1
        # the activity is not found... no cluster found
        if len(cluster_found_complex) == 0 and found == 0:
            confusion_matrix[dict_activity_index_colour[act_label]][-1] += 1

        if overlap_interval < 0:
            overlap_interval = 0
            overlap_interval_max = 0
        if not act_label in null_label:
            p = overlap_interval / (act_interval[1] - act_interval[0])
            percentage_coverage.append(int(p * 100 + 0.5) / 100.0)
            percentage_coverage_max.append(float(overlap_interval_max) / (act_interval[1] - act_interval[0]))
            if len(cluster_found_complex) > 0:
                clusters_during_activities_no_null.extend(cluster_found_complex)
                average_fragmentation_activities_same_color.append(
                    len([s for s in cluster_found_complex if not s[0] in null_label]) + found)
            if len(clusters_during_activities_dif_color_no_null) > 0:
                avg_rate = len([s for s in clusters_during_activities_dif_color_no_null if not s in used_clusters]) + found
                if avg_rate > 0: average_fragmentation_activities_diff_color.append(avg_rate)
                used_clusters.extend(clusters_during_activities_dif_color_no_null)

        confusion_matrix_detailed[dict_activity_index_colour[act_label]][-1] = act_interval[1] - act_interval[0] - sum(
            confusion_matrix_detailed[dict_activity_index_colour[act_label]])

    count_activities_not_found = percentage_coverage.count(0)

    couter_data_samples_no_null = []
    for i, act in enumerate(activities_not_null[:-1]):
        couter_data_samples_no_null.append(int(act[1][1] - act[1][0]))
    couter_data_samples_no_null.append(int(activities_not_null[-1][1][1] - activities_not_null[-1][1][0]))

    accuracy, recall, precision, f_measure = calculate_accuracy(confusion_matrix_detailed, couter_data_samples_no_null)

    if len(average_fragmentation_activities_same_color) == 0:
        average_fragmentation_activities_same_color.append([0])

    if len(average_fragmentation_activities_diff_color) == 0:
        average_fragmentation_activities_diff_color.append([0])

    h_accuracy, h_recall, h_precision, h_f_measure = calcualte_accuracy_hungarian(hungarian_matrix, couter_data_samples_no_null)

    if VISUALIZATION:
        print "\nEVALUATION:"
        print "\t\t\Accu\tF-meas\tFragmentation"
        print "Supervised:\t" + str(int((accuracy * 100) + 0.5) / 100.0) + "\t" + str(int((f_measure * 100) + 0.5) / 100.0) + "\t" + str(np.mean(average_fragmentation_activities_diff_color))
        print "Unsupervised:\t" + str(int((h_accuracy * 100) + 0.5) / 100.0) + "\t" + str(int((h_f_measure * 100) + 0.5) / 100.0) + "\t" + str(np.mean(average_fragmentation_activities_diff_color))

        print "\nNumber of not found activities (supervised identification): " + str(count_activities_not_found) + " out of: " + str(len(activities_not_null))
        print "Number of not found activities (unsupervised discovery ):" + str(h_recall.count(0)) + " out of: " + str(len(activities_not_null))

        # deal with overlapping clusters
        clusters_overlaping = []
        clusters_cleared = []
        color_overlaping = []
        color_cleared = []
        change = 0
        for i, cluster in enumerate(cluster_segments[:-1]):
            if change == 1:
                change = 0
                continue
            if cluster[0] + cluster[1] > cluster_segments[i + 1][0]:
                if change == 0:
                    clusters_cleared.append(cluster)
                    color_cleared.append(cluster_colors_set[i])
                    clusters_overlaping.append(cluster_segments[i + 1])
                    color_overlaping.append(cluster_colors_set[i + 1])
                    change = 1
                else:
                    clusters_cleared.append(cluster)
                    color_cleared.append(cluster_colors_set[i])
                    change = 0
            else:
                clusters_cleared.append(cluster)
                color_cleared.append(cluster_colors_set[i])
                change = 0
        if len(cluster_segments) > len(clusters_cleared) + len(clusters_overlaping):
            clusters_cleared.append(cluster_segments[-1])
            color_cleared.append(cluster_colors_set[-1])

        for i, cluster in enumerate(clusters_cleared[:-1]):
            if cluster[0] + cluster[1] > clusters_cleared[i + 1][0]:
                a = 5

        # visualize the gant chart
        fig3 = plt.figure(3)
        ax3 = fig3.add_subplot(111)

        overlapping = False
        if overlapping == True:
            ax3.broken_barh(activity_segments, (7, 2), facecolors=colors_set)
            ax3.broken_barh(clusters_cleared, (4, 2), facecolors=color_cleared)
            ax3.broken_barh(clusters_overlaping, (1, 2), facecolors=color_overlaping)

            ax3.set_ylim(0, 10)
            ax3.set_xlim(0, len(dataAll))
            ax3.set_xlabel('seconds since start')
            ax3.set_yticks([2, 5, 8])
            ax3.set_yticklabels(['Overlapping', 'Clusters', 'Activities'])
            ax3.grid(True)

        else:
            ax3.broken_barh(activity_segments, (4, 2), facecolors=colors_set)
            ax3.broken_barh(cluster_segments, (1, 2), facecolors=cluster_colors_set)

            ax3.set_ylim(0, 7)
            ax3.set_xlim(0, len(dataAll))
            ax3.set_xlabel('seconds since start')
            ax3.set_yticks([2, 5])
            ax3.set_yticklabels(['Clusters', 'Activities'])
            ax3.grid(True)

        for i, seg in enumerate(activity_segments):
            if (seg[1] > 150):
                if activity_order[i][0] in null_label:
                    continue
                a = int(seg[0]) + int(seg[1]) / 2.0 - 100
                if int(activity_order[i][0]) >9:
                    a-= 70
                ax3.text(a, 5, "A" + str(int(activity_order[i][0])), size = 12)


        for i, seg in enumerate(cluster_segments):
            if (seg[1]>180):
                a = int(seg[0]) + int(seg[1]) / 2.0 - 100
                if ignore_cluster == True and i >9:
                    a-= 70
                elif int(cluster_array[i][0]) > 9:
                    a -= 70
                if ignore_cluster == True:
                    ax3.text(a, 2, "C"+str(int(i+1)), size = 10)
                else:
                    ax3.text(a, 2, "C"+str(int(cluster_array[i][0])), size=11)
        plt.show()

    result = [h_accuracy, h_f_measure, 1.0 - float(h_recall.count(0)) / len(activities_not_null),
                            1.0 / np.mean(average_fragmentation_activities_diff_color),
                            accuracy, f_measure, 1.0 - float(count_activities_not_found) / len(activities_not_null),
                            np.mean(average_fragmentation_activities_same_color),
                            np.mean(average_fragmentation_activities_diff_color),
                            len(clusters_during_activities_no_null) / float(len(activities_not_null)),
                            len(clusters_during_activities_dif_color_no_null) / float(len(activities_not_null)),
                            count_activities_not_found, h_recall.count(0), n_clusters_, len(activities_not_null)]

    return confusion_matrix_detailed, hungarian_matrix, result