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

using namespace pcl;


/**
 * @brief The ply_segmentation class provides functionalities for point cloud segmentation
 * and cluster extraction, including visualization and transformation utilities.
 */
class ply_segmentation {
public:
    ply_segmentation() = default;

    /**
     * @brief segmentAndExtractClusters
     * @param cloud
     * @return
     */
    std::vector<pcl::PointIndices> segmentAndExtractClusters(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);

    /**
     * @brief extractLargestCluster
     * @param cluster_indices
     * @param cloud
     * @return
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr extractLargestCluster(const std::vector<pcl::PointIndices>& cluster_indices, const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);

    /**
     * @brief visualizePointCloud
     * @param cloud
     * @param cluster_indices
     */
    void visualizePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::vector<pcl::PointIndices>& cluster_indices);

    /**
     * @brief extractCluster
     * @param cloud
     * @param cluster
     * @return
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr extractCluster(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const pcl::PointIndices& cluster);


    /**
     * @brief transformCloud
     * @param cloud
     * @param reference_point
     * @return
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const pcl::PointXYZ& reference_point);

    /**
     * @brief subtractPointClouds
     * @param cloudA
     * @param cloudB
     * @param searchRadius
     * @return
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr subtractPointClouds(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudA, const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudB, float searchRadius);
};

#endif // PLY_SEGMENTATION_H
