#ifndef CLUSTERS_H
#define CLUSTERS_H

#include "3d_system_global.h"
#include <Eigen/Geometry> // For Eigen::Quaternionf
#include <Eigen/Dense> // For Eigen::Vector4f

struct ClusterInfo {
    Eigen::Vector4f centroid;
    Eigen::Vector3f dimensions;
    Eigen::Vector3f eulerAngles;
    Eigen::Quaternionf orientation;
    int clusterSize;
    int clusterId;
};

#endif // CLUSTERS_H
