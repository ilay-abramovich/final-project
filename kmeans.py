import math
import sys
import argparse

DEFAULT_ITER = 400

class centroid:
    def __init__(self, vector, points = set()):
        self.vector = vector
        self.points = points
        self.dimention = len(vector)
    
    def update_vector(self):
        if len(self.points) == 0:
            return
        for i in range(self.dimention):
            self.vector[i] = sum([p.vector[i] for p in self.points]) / len(self.points)
        return
    
    def add_point(self, p):
        if p.centroid:
            p.centroid.points.remove(p)
        p.centroid = self
        self.points.add(p)
        return

class datapoint:
    def __init__(self, vector, centroid = None):
        self.vector = vector
        self.centroid = centroid
        self.dimention = len(vector)

    def distance(self, other):
        if len(other.vector) != len(self.vector):
            print("Unequal vector lengths")
            return -1
        else:
            return math.sqrt(sum([(self.vector[i] - other.vector[i]) ** 2 for i in range(self.dimention)]))

def parse_input(vectors):
    vectors = [[float(d) for d in v.split(',')] for v in vectors]
    all_points = [datapoint(v) for v in vectors]
    return all_points


def cluster(k, datapoints, iter, epsilon = 0.001):
    centroids = [centroid(v.vector.copy(), set([])) for v in datapoints[:k]]
    below_threshold = True
    for i in range(iter):
        for p in datapoints:
            new_center = min(centroids, key=lambda c: p.distance(c))
            new_center.add_point(p)
            below_threshold &= (p.distance(p.centroid) <= epsilon)
            continue
        if below_threshold:
            return centroids
        for c in centroids:
            c.update_vector()
    return centroids

def main(k, data, iter='400'):
    if not(k.isdigit() and iter.isdigit()):
        print(k)
        print("Invalid input")
        exit(1)
    k, iter = int(k), int(iter)
    if iter > 1000 or iter < 1:
        print("Incorrect maximum iteration!")
        return 0
    data = parse_input(data)
    if len(data) < k:
        print("Incorrect number of clusters!")
        return
    
    clusters = cluster(k, data, iter)
    for c in clusters:
        print(",".join(['%.4f' % d for d in c.vector]))
    return

if __name__ == '__main__':
    data = sys.stdin.readlines()
    if len(sys.argv) < 3:
        main(sys.argv[1], data)
    else:
        main(sys.argv[1], data, sys.argv[2])

