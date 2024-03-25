#ifndef PLY_SEGMENTATION_H
#define PLY_SEGMENTATION_H

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/common.h>
#include <pcl/common/eigen.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/segmentation/region_growing.h>

#include <pcl/filters/radius_outlier_removal.h>
#include <random>
#include "clusters.h"

using namespace pcl;


/**
 * @brief The ply_segmentation class provides functionalities for point cloud segmentation
 * and cluster extraction, including visualization and transformation utilities.
 */
class ply_segmentation {
public:
    ply_segmentation() = default;

    /**
     * @brief Function to segment and extract clusters from a point cloud. The function uses Euclidean clustering to segment the cloud.
     * @param cloud The point cloud to segment.
     * @return A vector of point indices representing the clusters.
     */
    std::vector<pcl::PointIndices> segmentAndExtractClusters(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);

    /**
     * @brief Function to extract the largest cluster from a point cloud.
     * @param cluster_indices The vector of point indices representing the clusters.
     * @param cloud The point cloud to extract the cluster from.
     * @return The largest cluster as a point cloud.
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr extractLargestCluster(const std::vector<pcl::PointIndices>& cluster_indices, const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);

    /**
     * @brief Function to visualize the point cloud and the extracted clusters.
     * @param cloud The point cloud to visualize.
     * @param cluster_indices The vector of point indices representing the clusters.
     * @return The visualizer object.
     */
    void visualizePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::vector<pcl::PointIndices>& cluster_indices);

    /**
     * @brief Function to visualize the point cloud and the extracted clusters.
     * @param cloud The point cloud to visualize.
     * @param clusters The vector of clusters to visualize.
     * @return The visualizer object.
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr extractCluster(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const pcl::PointIndices& cluster);


    /**
     * @brief Function to transform a point cloud to align with the specified reference point.
     * @param cloud The point cloud to transform.
     * @param reference_point The reference point to align the cloud with.
     * @return The transformed point cloud.
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const pcl::PointXYZ& reference_point);

    /**
     * @brief Function to subtract two point clouds. Usually cloudA is the tray which boxes and cloudB is just tray.
     * @param cloudA The first point cloud.
     * @param cloudB The second point cloud.
     * @param searchRadius The search radius for the subtraction.
     * @return The subtracted point cloud.
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr subtractPointClouds(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudA, const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudB, float searchRadius);

    /**
     * @brief Function to extract the locations of the clusters.
     * @param cloud The point cloud to extract the locations from.
     * @param cluster_indices The vector of point indices representing the clusters.
     * @return A vector of ClusterInfo objects containing the bounding box information for each cluster.
     */
    std::vector<ClusterInfo> extractLocations(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::vector<pcl::PointIndices>& cluster_indices);

    /**
     * @brief Function to find the reference point in the point cloud.
     * @param cloud The point cloud to find the reference point in.
     * @return The reference point.
     */
    PointXYZ findReferencePoint(const PointCloud<PointXYZ>::Ptr& cloud);

};

#endif // PLY_SEGMENTATION_H
