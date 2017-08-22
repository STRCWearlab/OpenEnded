import copy
import heapq
import math
import operator
import numpy as np
import scipy
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def kernel_linear(x, y):
    return scipy.dot(x, y)


def kernel_poly(x, y, a=1.0, b=1.0, p=2.0):
    return (a * scipy.dot(x, y) + b) ** p


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
    # if not
    # return kernel(x,x)-2*kernel(x,y)+kernel(y,y)

kernel = kernel_normalise(kernel_gauss)

# the time past which you cannot add an instance to the cluster... activity frequency?
deltaT = 5 #bigger number bigger clusters, tends to combine small clusters with big ones

# if the cluster is older than memoryDelta, then remove it from the currentClusters and put it in the allClsuters list
memoryDelta = deltaT + 1  #bigger number smaler clusters, lots of empty space... only sure clusters

#the number of current clusters... pool of clusters
num_clusterss = 4 #bigger number scattered clusters, lots of empty space... if you increase this, also increase the memory parameters
threshold_cluster_size = 10


class OnlineVariance(object):
    """
    Welford's algorithm computes the sample variance incrementally.
    """

    def __init__(self, iterable=None, ddof=1):
        self.ddof, self.n, self.mean, self.M2, self.std = ddof, 0, 0.0, 0.0, 0.0

    def std_calc(self, datum):
        self.n += 1
        self.delta = datum - self.mean
        self.mean += self.delta / self.n
        self.M2 += self.delta * (datum - self.mean)
        self.variance = self.M2 / (self.n - self.ddof)
        self.std = np.sqrt(self.variance)

    def merge_std(self, new):
        n_mean = (self.n * self.mean + new.n * new.mean) / (self.n + new.n)
        self.variance = (self.n * self.std * self.std + new.n * new.std * new.std + self.n * (self.mean - n_mean) * (
        self.mean - n_mean) + new.n * (new.mean - n_mean) * (new.mean - n_mean)) / (self.n + new.n)
        self.std = np.sqrt(self.variance)
        self.M2 = self.variance * (self.n + new.n - self.ddof)
        self.n += new.n
        self.mean = n_mean

class Cluster(object):
    def __init__(self, a, timestamp):
        self.center = a
        self.size = kernel(a, a)
        self.timestampEnd = timestamp
        self.timestampStart = timestamp
        self.firstPoint = a
        self.endPoint = a
        self.num_points = 1
        self.STD = OnlineVariance(ddof=0)
        self.STD.std_calc(kernel(a, a))

    def add(self, e, timestampNew):
        self.size += kernel(self.center, e)
        self.center += (e - self.center) / self.size
        self.timestampEnd = timestampNew
        self.endPoint = e
        self.num_points += 1
        self.STD.std_calc(kernel(self.center, e))

    def merge(self, c):
        self.center = (self.center * self.size + c.center * c.size) / (self.size + c.size)
        self.size += c.size
        self.num_points += c.num_points
        self.num_points -= 1
        self.STD.merge_std(c.STD)

        if self.timestampEnd < c.timestampEnd:
            self.timestampEnd = c.timestampEnd
            self.endPoint = c.endPoint

        if self.timestampStart > c.timestampStart:
            self.timestampStart = c.timestampStart
            self.firstPoint = c.firstPoint

    def resize(self, dim):
        extra = scipy.zeros(dim - len(self.center))
        self.center = scipy.append(self.center, extra)

    def __str__(self):
        return "Cluster( %s, %f, %f , %f )" % (self.center, self.size, self.timestampStart, self.timestampEnd)


class Dist(object):
    """this is just a tuple,
    but we need an object so we can define cmp for heapq"""

    def __init__(self, x, y, d):
        self.x = x
        self.y = y
        self.d = d

    def __cmp__(self, o):
        return cmp(self.d, o.d)

    def __str__(self):
        return "Dist(%f)" % (self.d)

p = 0
class OnlineCluster(object):
    def __init__(self, N):
        """N-1 is the largest number of clusters I can find Higher N makes me slower"""
        self.n = 0
        self.N = N
        self.allClusters = []
        self.currentClusters = []
        # max number of dimensions we've seen so far
        self.dim = 0

        # cache inter-cluster distances
        self.dist = []

    def resize(self, dim):
        for c in self.currentClusters:
            c.resize(dim)
        self.dim = dim

    def cluster(self, e, timestamp):
        #check if there are old clusters and move them to memory
        possibleMerge = []
        for clusterI in self.currentClusters:
            # if the cluster is older than memoryDelta, then remove it from the currentClusters and put it in the allClsuters list
            if timestamp - clusterI.timestampEnd >= memoryDelta:
                self.currentClusters.remove(clusterI)
                self.removedist(clusterI)

                #merging clusters... TODO
                if len(self.allClusters) > 0:
                    min_distance = kernel(clusterI.center, self.allClusters[-1].center)
                    closest_cluster = copy.deepcopy(self.allClusters[-1])
                    for clusterR in reversed(self.allClusters):
                        if clusterI.timestampEnd - clusterR.timestampEnd < memoryDelta:
                            possibleMerge.append (clusterR)
                            distance = kernel(clusterI.center, clusterR.center)
                            if min_distance > distance:
                                min_distance = distance
                                closest_cluster = clusterR
                        else:
                            break

                    if (abs(min_distance-closest_cluster.STD.mean) < p*closest_cluster.STD.std) or (abs(min_distance-clusterI.STD.mean) < p*clusterI.STD.std):
                        for i in reversed(self.allClusters):
                            if i.STD.std == closest_cluster.STD.std:
                                self.allClusters.remove(i)
                                break
                        closest_cluster.merge(clusterI)
                        self.allClusters.append(closest_cluster)
                    else:
                        self.allClusters.append(clusterI)
                else:
                    self.allClusters.append(clusterI)

        if len(e) > self.dim:
            self.resize(len(e))

        if len(self.currentClusters) > 0:
            # compare new points to each existing cluster
            c = [(i, kernel_dist(x.center, e)) for i, x in enumerate(self.currentClusters)]
            closest = self.currentClusters[min(c, key=operator.itemgetter(1))[0]]
            #if the point is older than deltaT, do not add it
            if (timestamp - closest.timestampEnd) < deltaT:
                closest.add(e, timestamp)
                # invalidate dist-cache for this cluster
                self.updatedist(closest)
            else:
                aaa = 5

        if len(self.currentClusters) >= self.N and len(self.currentClusters) > 1:
            # merge closest two clusters
            m = heapq.heappop(self.dist)
            try:
                self.currentClusters.remove(m.y)
                self.removedist(m.y)
                m.x.merge(m.y)
                self.updatedist(m.x)
            except:
                print m.y

        # make a new cluster for this point
        newc = Cluster(e, timestamp)
        self.currentClusters.append(newc)
        self.updatedist(newc)
        self.n += 1

    def removedist(self, c):
        """invalidate intercluster distance cache for c"""
        r = []
        for x in self.dist:
            if x.x == c or x.y == c:
                r.append(x)
        for x in r: self.dist.remove(x)
        heapq.heapify(self.dist)

    def updatedist(self, c):
        """Cluster c has changed, re-compute all intercluster distances"""
        self.removedist(c)

        for x in self.currentClusters:
            if x == c: continue
            d = kernel_dist(x.center, c.center)
            t = Dist(x, c, d)
            heapq.heappush(self.dist, t)

    def trimclusters(self):
        """Return only clusters over threshold"""
        clusters = self.allClusters + self.currentClusters
        #t = scipy.mean([x.size for x in filter(lambda x: x.size > 0, clusters)]) * 0.5
        t = threshold_cluster_size
        #print "Threshold: " + str(t)

        return filter(lambda x: x.num_points >= t, clusters)
        #return filter(lambda x: x.num_points >= t and x.num_points / x.size < 0.73, clusters)

activity_means = [[3.0, 5.0], [7.0, 7.0], [5.0, 5.0]]
