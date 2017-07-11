import copy
import time
import matplotlib.pyplot as plt
import numpy as np
import six
from hmmlearn import hmm
from matplotlib import colors
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import Online_temporal_clustering as OTC
import sklearn


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def transform_activities (activity_list):
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


#Returns the mean values (centers) of each activity (if activity is repeated, then it is considered as the same activity)...
# ToDo: it should not be like this... each repetition is a separate activity
def get_activity_means(dataAll):
    keys = list(set(dataAll[:, [3]].T[0]))
    activity_array_temp = []
    for x in keys:
        activity_array_temp.append(dataAll[np.logical_or.reduce([dataAll[:, -1] == x])])

    activity_means = []
    for item in activity_array_temp:
        activity_means.append([np.mean(item[:, 0]), np.mean(item[:, 1])])
    return activity_means


###########################################
# parameters
np.random.seed(0)

OTC.sigmaP= 0.5
OTC.kernel = OTC.kernel_normalise(OTC.kernel_gauss)

# the time past which you cannot add an instance to the cluster... activity frequency?
OTC.deltaT = 5 #bigger number bigger clusters, tends to combine small clusters with big ones

# if the cluster is older than memoryDelta, then remove it from the currentClusters and put it in the allClsuters list
OTC.memoryDelta = OTC.deltaT + 10  #bigger number smaler clusters, lots of empty space... only sure clusters

#the number of current clusters... pool of clusters
OTC.num_clusterss = 4 #bigger number scattered clusters, lots of empty space... if you increase this, also increase the memory parameters
OTC.threshold_cluster_size = 10

plot_all = False
null_label = 0
###########################################

if __name__ == '__main__':
    #data_features = np.loadtxt('_data/data_features.csv', delimiter=';')
    data_features = np.loadtxt('_data/data_features_null_overlap.csv', delimiter='\t')
    data_features = sorted(data_features, key=lambda a_entry: a_entry[0])

    data_array = np.array(data_features)
    data_array[:, [3, 5]] = sklearn.preprocessing.scale(data_array[:, [3, 5]])

    X = data_array[:, [3, 5]]
    y = data_array[:, -1].astype(int)

    dataAll = np.column_stack((X, data_array[:, 0], data_array[:, -1]))


    ###############################################
    points = dataAll[:, [0, 1]]
    timestamps = dataAll[:, [2]]
    n = len(points)
    start = time.time()

    c = OTC.OnlineCluster(OTC.num_clusterss)
    for ind1, point in enumerate(copy.deepcopy(points)):
        c.cluster(point, timestamps[ind1])
    clusters = c.trimclusters()
    print "I clustered %d points in %.2f seconds and found %d clusters." % (n, time.time() - start, len(clusters))

    activity_means = get_activity_means(dataAll)
    activities_set = list(set(dataAll[:, [3]].T[0]))
    dict_activity_index_colour = dict(zip(activities_set, np.arange(len(activities_set))))  # {1:0, 2:1, 6:2, 32:3}

    # activities_set_no_null = [x for x in activities_set_no_null if x != null_label]
    # dict_activity_index = dict(zip(activities_set_no_null, np.arange(len(activities_set_no_null))))


    #Plot 1st Figure
    if plot_all == True:
        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter([x[0] for x in dataAll], [x[1] for x in dataAll], [x[2] for x in dataAll])
        for cluster in clusters:
            # visualize the arrows - start and end point of each cluster
            a = Arrow3D([cluster.center[0], cluster.center[0]], [cluster.center[1], cluster.center[1]], [cluster.timestampStart, cluster.timestampEnd], arrowstyle="-|>", mutation_scale=20, lw=5, color="r")
            ax.add_artist(a)
    cx = [x.center[0] for x in clusters]
    cy = [y.center[1] for y in clusters]
    cz = [z.timestampEnd for z in clusters]
    #if plot_all == True: ax.plot(cx, cy, cz, "ro")

    color = list(six.iteritems(colors.cnames))

    #Plot 2nd Figure
    j = 0
    handles = []
    colors_clusters = []
    if plot_all == True:
        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.scatter(cx, cy, cz, marker="x", s=150, linewidths=5, zorder=100)
        ax2.set_ylabel('Feature 2')
        ax2.set_xlabel('Feature 1')
        ax2.set_zlabel('Time')
        ax2.legend(handles, [str(activities_set) +  "NULL"])
        # visualize the arrows - start and end point of each cluster
        for cluster in clusters:
            a = Arrow3D([cluster.center[0], cluster.center[0]], [cluster.center[1], cluster.center[1]],[cluster.timestampStart, cluster.timestampEnd], arrowstyle="-|>", mutation_scale=20, lw=5, color="r")
            ax2.add_artist(a)
    for point in points:
        if plot_all == True:
            handles.append(ax2.scatter(point[0], point[1], dataAll[j][2], c=color[int(dict_activity_index_colour[dataAll[j][3]])]))
            colors_clusters.append(int(dict_activity_index_colour[dataAll[j][3]]))
        j += 1

    cluster_segments = []
    cluster_segments_complex = []
    cluster_colors_set = []
    cluster_array = []
    ratios = []

    for cluster in clusters:
        activity_means = np.asarray(activity_means)
        min_distance = OTC.kernel_dist(cluster.center, activity_means[0])
        min_index = index_temp = 0
        for mean in activity_means:
            distance_temp = OTC.kernel_dist(cluster.center, mean)
            if min_distance > distance_temp:
                min_distance = distance_temp
                min_index = index_temp
            index_temp+=1
        cluster_segments_complex.append(((int(cluster.timestampStart), (int(cluster.timestampEnd) - int(cluster.timestampStart))),color[int(min_index)][0]))
        #cluster_colors_set.append(color[int(min_index)][0])
        cluster_array.append([min_index, [int(cluster.timestampStart), int(cluster.timestampEnd)]])

        ratios.append([min_index, int(cluster.timestampStart), int((cluster.num_points / cluster.size * 100) + 0.5) / 100.0])
        if null_label == -1:
            print "Cluster start-end:\t" + str(cluster.timestampStart) + "-" + str(cluster.timestampEnd) + "\t\t" + "Cluster size:\t" + str(cluster.size) + "\tCluster num points:\t" + str(cluster.num_points) + '\t Ratio:' + str(cluster.num_points / cluster.size)
        else:
            if min_index == null_label: print "Cluster start-end:\t" + str(cluster.timestampStart) + "-" + str(cluster.timestampEnd) + "\t\t" + "Cluster size:\t" + str(cluster.size) + "\tCluster num points:\t" + str(cluster.num_points) + '\t Ratio:' + str(cluster.num_points / cluster.size) + "\tNULLLLL"
            else: print "Cluster start-end:\t" + str(cluster.timestampStart) + "-" + str(cluster.timestampEnd) + "\t\t" + "Cluster size:\t" + str(cluster.size) + "\tCluster num points:\t" + str(cluster.num_points) + '\t Ratio:' + str(cluster.num_points / cluster.size)

    ratios  = sorted(ratios, key=lambda x: (-x[1], x[0]))
    print "\nActivity, Movement_Ratio: \t" + str(ratios)
    cluster_segments_complex =  sorted(cluster_segments_complex, key=lambda x: x[0])
    for cs in cluster_segments_complex:
        cluster_segments.append(cs[0])
        cluster_colors_set.append(cs[1])

    #validation, performance
    cluster_colors_set = tuple(cluster_colors_set)
    t = transform_activities(dataAll[:, [3]])
    activity_segments = t[0]
    activity_order = t[1]
    activity_array = t[2]

    colors_set = []
    for ind in activity_order:
        colors_set.append(color[int(dict_activity_index_colour[int(ind)])][0])
    colors_set = tuple(colors_set)

    border = 5
    counter_activities = 0
    percentage_coverage = []
    percentage_coverage_max = []
    weights = []

    confusion_matrix = np.zeros((len(activities_set), len(activities_set) + 1))
    count_clusters_during_null = 0

    for activity_item in activity_array:
        act_label = activity_item[0]
        act_interval = activity_item[1]

        if act_interval[1] == act_interval[0]: continue
        counter_activities +=1
        if not act_label == null_label: weights.append(act_interval[1] - act_interval[0])
        overlap_interval = 0
        overlap_interval_max = 0
        cluster_found = []
        cluster_found_complex = []

        for cluster_item in cluster_array:
            current_overlap_interval = 0
            clus_label = cluster_item[0]
            clus_interval = cluster_item[1]
            #if the cluster starts or ends inside the activity interval
            if (clus_interval[0]>=act_interval[0] - border and clus_interval[0]<=act_interval[1]+ border) or (clus_interval[1]>=act_interval[0] - border and clus_interval[1]<=act_interval[1] + border):
                if clus_label == dict_activity_index_colour[act_label] or clus_label == null_label:
                    cluster_found.append(clus_interval)
                    cluster_found_complex.append(cluster_item)
                    start = max(clus_interval[0], act_interval[0])
                    end = min(clus_interval[1], act_interval[1])
                    overlap_interval += float(end - start)
                    current_overlap_interval = float(end - start)
                    overlap_interval_max = overlap_interval
                else:
                    #if current_overlap_interval < (0.1 * act_interval[1] - act_interval[1]):
                    continue
            else:
                continue
        if overlap_interval<0:
            overlap_interval = 0
            overlap_interval_max = 0

        if len(cluster_found) > 1:
            #find the biggest cluster inside the activity
            max_value = 0
            cluster_found_complex2 = copy.deepcopy(cluster_found_complex)
            for cl in cluster_found_complex2:
                cl[1][0] = max(cl[1][0], act_interval[0])
                cl[1][1] = min(cl[1][1], act_interval[1])
                if cl[1][1] - cl[1][0] >= max_value:
                    max_value = cl[1][1] - cl[1][0]
                    max_act = cl[0]
            #fill the confusion matrix
            confusion_matrix[dict_activity_index_colour[act_label]][max_act] += 1
            overlap_interval_max = max_value

            #deal with overlapping clusters
            overlap_interval = 0
            cluster_found2 = copy.deepcopy(cluster_found)
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
                    overlap_interval += b-a
                    a, b = c, d
            overlap_interval += float(b - a)

        elif len(cluster_found_complex) == 1:
            confusion_matrix[dict_activity_index_colour[act_label]][cluster_found_complex[0][0]] += 1
        #no clusters were found
        else:
            #check if there is a cluster bigger than this activity (bursts of the same activity covered by the same cluster)
            found = 0
            for clusterrr in cluster_array:
                clus_label = clusterrr[0]
                clus_interval = clusterrr[1]
                if act_interval[0] > clus_interval[0] and  act_interval[1] < clus_interval[1]:
                    confusion_matrix[dict_activity_index_colour[act_label]][clus_label] += 1
                    if clus_label == dict_activity_index_colour[act_label] or clus_label == null_label:
                        overlap_interval = act_interval[1] - act_interval[0]
                        overlap_interval_max = act_interval[1] - act_interval[0]
                    found = 1
            #the activity is not found... no cluster found
            if found == 0:
                confusion_matrix[dict_activity_index_colour[act_label]][-1] += 1
                if not act_label == null_label: print "Activity not found..." + str(activity_item) + "\tthe size of the activity is: " + str(act_interval[1] - act_interval[0])

        if overlap_interval<0:
            overlap_interval = 0
            overlap_interval_max = 0
        if not act_label == null_label:
            percentage_coverage.append(overlap_interval/(act_interval[1] - act_interval[0]))
            percentage_coverage_max.append(float(overlap_interval_max) / (act_interval[1] - act_interval[0]))
    weights = [x / float(sum(weights)) for x in weights]
    percentage_coverage_weighted = [x*y for x, y in zip (percentage_coverage, weights)]
    percentage_coverage_weighted_max = [x * y for x, y in zip(percentage_coverage_max, weights)]

    count_activities_not_found = percentage_coverage.count(0)
    #count_activities_not_found = np.sum(confusion_matrix[:,-1][1:])
    count_clusters_during_null = len(clusters) - (len(percentage_coverage) - count_activities_not_found)


    print "\nPercentage coverage for each activity\t" + str(percentage_coverage)
    print "Percentage coverage for each activity biggest cluster\t" + str(percentage_coverage_max)

    print "\nAveraged percentage coverage\t" + str(sum(percentage_coverage)/len(percentage_coverage))
    print "Averaged percentage coverage biggest cluster\t" + str(sum(percentage_coverage_max) / len(percentage_coverage_max))

    print "\nWeighted Averaged coverage\t" + str(sum(percentage_coverage_weighted))
    print "Weighted Averaged coverage biggest cluster\t" + str(sum(percentage_coverage_weighted_max))

    print "\nNumber of not found activities: " + str(count_activities_not_found)
    print "Number of clusters during NULL: " + str(count_clusters_during_null)

    print "\nConfusion Matrix: activities/predictions"
    print str(activities_set) + " NotFound ]"
    print confusion_matrix
    print "Total activities: " + str(counter_activities)

    # visualize the gant chart
    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot(111)

    ax3.broken_barh(activity_segments, (13, 4), facecolors=colors_set)
    ax3.broken_barh(cluster_segments, (8, 4), facecolors=cluster_colors_set)

    ax3.set_ylim(5, 20)
    ax3.set_xlim(0, len(dataAll))
    ax3.set_xlabel('seconds since start')
    ax3.set_yticks([10, 15])
    ax3.set_yticklabels(['Clusters', 'Activities'])
    ax3.grid(True)

    plt.show()
