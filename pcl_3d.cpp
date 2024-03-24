#include "pcl_3d.h"
#include <iostream>
#include <stdexcept>

//function to calibrate tray
Eigen::Vector3f PCL_3D::calibrateTray(const std::string& filePath, float height)
{
    auto cloudOpt = processor->loadCloud(filePath);
    if (!cloudOpt) {
        std::cerr << "Failed to read point cloud." << std::endl;
        throw std::runtime_error("Failed to read point cloud.");
    }
    auto cloud = cloudOpt.value(); // Dereference std::optional

    // Step 1: Invert the point cloud
    if (!processor->invertPointCloud(*cloud)) {
        std::cerr << "Inverting point cloud failed." << std::endl;
        throw std::runtime_error("Inverting point cloud failed.");
    }

    // Step 2: Apply Z PassThrough Filter
    if (!processor->applyPassThroughZOnly(*cloud, -height, 0)) {
        std::cerr << "Applying PassThrough Z filter failed." << std::endl;
        throw std::runtime_error("Applying PassThrough Z filter failed.");
    }

    // Step 3: Apply Voxel Grid Filter
    if (!processor->applyVoxelGridFilter(*cloud, 0.01f)) { // Example leaf size
        std::cerr << "Applying Voxel Grid filter failed." << std::endl;
        throw std::runtime_error("Applying Voxel Grid filter failed.");
    }

    // Step 4: Apply MLS Surface Reconstruction
    if (!processor->applyMLSSurfaceReconstruction(*cloud, 0.01f)) { // Example search radius
        std::cerr << "Applying MLS Surface Reconstruction failed." << std::endl;
        throw std::runtime_error("Applying MLS Surface Reconstruction failed.");
    }

    // Step 5: Remove Outliers
    if (!processor->removeOutliers(*cloud, 5, 1.7)) { // Example meanK and stddevMulThresh
        std::cerr << "Removing outliers failed." << std::endl;
        throw std::runtime_error("Removing outliers failed.");
    }

    // Step 6: Segment and Extract Clusters
    auto cluster_indices = segmentation->segmentAndExtractClusters(cloud);

    // Extract the largest cluster and find the reference point
    auto largest_cluster = segmentation->extractLargestCluster(cluster_indices, cloud);
    pcl::PointXYZ reference_point = segmentation->findReferencePoint(largest_cluster);

    reference_point.z += 5;

    std::cout << "Reference point: " << reference_point.x << " " << reference_point.y << " " << reference_point.z << std::endl;

    //transform the point cloud to the reference point
    segmentation->transformCloud(cloud, reference_point);

    //Convert the reference point to Eigen::Vector3f
    Eigen::Vector3f reference_point_eigen(reference_point.x, reference_point.y, reference_point.z);

    std::cout << "Reference point (Eigen): " << reference_point_eigen.transpose() << std::endl;

    return reference_point_eigen;

}
