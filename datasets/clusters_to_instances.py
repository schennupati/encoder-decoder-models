#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 20:53:38 2019

@author: sumche
"""
import numpy as np
import matplotlib.pyplot as plt


class Point():
    def __init__(self, x_dist, y_dist, row, col):
        self._x_dist = x_dist
        self._y_dist = y_dist
        self._row = row
        self._col = col
        self._x_center = -x_dist + col
        self._y_center = -y_dist + row
        #self._dist_to_center = np.sqrt((col - self._x_center)**2 + (row - self._y_center)**2)
        self.cd = None  # core distance
        self.rd = None  # reachability distance
        self.processed = False

    def distance(self, point):
        return np.sqrt((point._x_center - self._x_center)**2 + (point._y_center - self._y_center)**2)


def pre_process(x_y_mask):
    point_list = []
    y_dist = x_y_mask[:, :, 0]
    x_dist = x_y_mask[:, :, 1]
    mask = x_y_mask[:, :, 2]
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if mask[row, col] == 0:
                continue
            new_point = Point(x_dist[row][col], y_dist[row][col], row, col)
            point_list.append(new_point)
    return point_list


class Optics():
    def __init__(self, pts_list, min_cluster_size, epsilon):
        self.pts = pts_list
        self.min_cluster_size = min_cluster_size
        self.max_radius = epsilon

    def _setup(self):
        for p in self.pts:
            p.rd = None
            p.processed = False
        self.unprocessed = [p for p in self.pts]
        self.ordered = []

    def _core_distance(self, point, neighbors):
        if point.cd is not None:
            return point.cd
        if len(neighbors) >= self.min_cluster_size - 1:
            sorted_neighbors = sorted([n.distance(point) for n in neighbors])
            point.cd = sorted_neighbors[self.min_cluster_size - 2]
        return point.cd

    def _neighbors(self, point):
        return [p for p in self.pts if p is not point and p.distance(point) <= self.max_radius]

    def _processed(self, point):
        point.processed = True
        self.unprocessed.remove(point)
        self.ordered.append(point)

    def _update(self, neighbors, point, seeds):
        # for each of point's unprocessed neighbors n...
        for n in [n for n in neighbors if not n.processed]:
            # find new reachability distance new_rd
            # if rd is null, keep new_rd and add n to the seed list
            # otherwise if new_rd < old rd, update rd
            new_rd = max(point.cd, point.distance(n))
            if n.rd is None:
                n.rd = new_rd
                seeds.append(n)
            elif new_rd < n.rd:
                n.rd = new_rd

    def run(self):
        self._setup()
        # for each unprocessed point (p)...
        while self.unprocessed:
            point = self.unprocessed[0]
            # mark p as processed
            # find p's neighbors
            self._processed(point)
            point_neighbors = self._neighbors(point)
            # if p has a core_distance, i.e has min_cluster_size - 1 neighbors
            if self._core_distance(point, point_neighbors) is not None:
                # update reachability_distance for each unprocessed neighbor
                seeds = []
                self._update(point_neighbors, point, seeds)
                # as long as we have unprocessed neighbors...
                while (seeds):
                    # find the neighbor n with smallest reachability distance
                    seeds.sort(key=lambda n: n.rd)
                    n = seeds.pop(0)
                    # mark n as processed
                    # find n's neighbors
                    self._processed(n)
                    n_neighbors = self._neighbors(n)
                    # if p has a core_distance...
                    if self._core_distance(n, n_neighbors) is not None:
                        # update reachability_distance for each of n's neighbors
                        self._update(n_neighbors, n, seeds)
        # when all points have been processed
        # return the ordered list
        return self.ordered

    def cluster(self, cluster_threshold):
        clusters = []
        separators = []
        for i in range(len(self.ordered)):
            this_i = i
            this_p = self.ordered[i]
            if this_p.rd != None:
                this_rd = this_p.rd
            else:
                this_rd = float('infinity')
            # use an upper limit to separate the clusters
            if this_rd > cluster_threshold:
                separators.append(this_i)
        separators.append(len(self.ordered))

        for i in range(len(separators) - 1):
            start = separators[i]
            end = separators[i + 1]
            if end - start >= self.min_cluster_size:
                clusters.append(Cluster(self.ordered[start:end]))
        return clusters


class Cluster:
    def __init__(self, points):
        self.points = points

    # --------------------------------------------------------------------------
    # calculate the centroid for the cluster
    # --------------------------------------------------------------------------

    def centroid(self):
        center = [sum([p._x_center for p in self.points]) / len(self.points),
                     sum([p._y_center for p in self.points]) / len(self.points)]
        return center

def get_color(num):
    return np.random.randint(0, 255, size=(3))


def to_rgb(bw_im):
    instances = np.unique(bw_im)
    instances = instances[instances != 0]
    rgb_im = [np.zeros(bw_im.shape, dtype=int), 
              np.zeros(bw_im.shape, dtype=int), 
              np.zeros(bw_im.shape, dtype=int)]
    for instance in instances:
        color = get_color(instance)
        rgb_im[0][instance == bw_im] = color[0]
        rgb_im[1][instance == bw_im] = color[1]
        rgb_im[2][instance == bw_im] = color[2]
    return np.stack([rgb_im[0],rgb_im[1],rgb_im[2]],axis=-1)
    #return np.concatenate([np.concatenate([np.expand_dims(rgb_im[0], -1), np.expand_dims(rgb_im[1], -1)], 2), np.expand_dims(rgb_im[2], -1)], 2)

def calc_clusters_img(raw_img):
    pts_list = pre_process(raw_img)
    min_cluster_size = 5
    epsilon = 5
    cluster_limit = 100
    op = Optics(pts_list, min_cluster_size, epsilon)
    _ = op.run()
    clusters = op.cluster(cluster_limit)
    new_img = np.zeros((raw_img.shape[0], raw_img.shape[1]))
    for i, cluster in zip(range(len(clusters)), clusters):
        for pt in cluster.points:
            new_img[pt._row, pt._col] = (i + 1)
    return to_rgb(new_img)