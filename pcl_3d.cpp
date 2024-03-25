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
    if (!processor->applyVoxelGridFilter(*cloud, 0.5f)) { // Example leaf size
        std::cerr << "Applying Voxel Grid filter failed." << std::endl;
        throw std::runtime_error("Applying Voxel Grid filter failed.");
    }

    // Step 4: Apply MLS Surface Reconstruction
    if (!processor->applyMLSSurfaceReconstruction(*cloud, 1.5f)) { // Example search radius
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

//function to find objects
std::vector<ClusterInfo> PCL_3D::findBoundingBox(const std::string& filePathBox,
                                        const std::string& filePathTray,
                                        const Eigen::Vector3f& referencePoint,
                                        const Eigen::Vector3f& prevLocation){

    //Check if prev location is empty, if empty use reference point for pass through filter
    pcl::PointXYZ minPt, maxPt;

    if(prevLocation.isZero()){
        minPt = {referencePoint.x() - 10, referencePoint.y() - 10, referencePoint.z() - 10};
        maxPt = {referencePoint.x() + 1000, referencePoint.y() + 1000, referencePoint.z() + 1000};
    }else{
        minPt = {prevLocation.x() - 100, prevLocation.y() - 100, prevLocation.z()};
        maxPt = {prevLocation.x() + 100, prevLocation.y() + 100, prevLocation.z() + 2500};
    }

    // Step 1: Load the point cloud with the object
    auto cloudOpt = processor->loadCloud(filePathBox);
    if (!cloudOpt) {
        std::cerr << "Failed to read point cloud." << std::endl;
        throw std::runtime_error("Failed to read point cloud.");
    }

    auto cloud = cloudOpt.value(); // Dereference std::optional

    // Step 2: Invert the point cloud
    if (!processor->invertPointCloud(*cloud)) {
        std::cerr << "Inverting point cloud failed." << std::endl;
        throw std::runtime_error("Inverting point cloud failed.");
    }

    // Step 3: Apply PassThrough Filter
    if (!processor->applyPassthroughFilter(*cloud, minPt, maxPt)) {
        std::cerr << "Applying PassThrough filter failed." << std::endl;
        throw std::runtime_error("Applying PassThrough filter failed.");
    }

    //Step 4: Load the point cloud with the tray
    auto cloudTrayOpt = processor->loadCloud(filePathTray);
    if (!cloudTrayOpt) {
        std::cerr << "Failed to read point cloud." << std::endl;
        throw std::runtime_error("Failed to read point cloud.");
    }

    auto cloudTray = cloudTrayOpt.value(); // Dereference std::optional

    // Step 5: Invert the point cloud
    if (!processor->invertPointCloud(*cloudTray)) {
        std::cerr << "Inverting point cloud failed." << std::endl;
        throw std::runtime_error("Inverting point cloud failed.");
    }

    // Step 6: Apply PassThrough Filter
    if (!processor->applyPassthroughFilter(*cloudTray, minPt, maxPt)) {
        std::cerr << "Applying PassThrough filter failed." << std::endl;
        throw std::runtime_error("Applying PassThrough filter failed.");
    }

    // Step 7: Subtract the tray from the object
    auto isolated_pcl = segmentation->subtractPointClouds(cloud, cloudTray, 20);
    std::cout << "Isolated pcl size: " << isolated_pcl->size() << std::endl;

    // Step 8: Segment and Extract Clusters
    auto cluster_indices = segmentation->segmentAndExtractClusters(isolated_pcl);

    // Transform the point cloud to align with the specified reference point
    //Convert the reference point to pcl::PointXYZ
    pcl::PointXYZ referencePointXYZ(referencePoint.x(), referencePoint.y(), referencePoint.z());
    auto transformed_cloud = segmentation->transformCloud(isolated_pcl, referencePointXYZ);

    //Step 9: Extract locations
    auto info = segmentation->extractLocations(transformed_cloud, cluster_indices);

    // Step 10: Visualize the results
    //segmentation->visualizePointCloud(transformed_cloud, cluster_indices);


    return info;
}

//function to transform point cloud to reference point
pcl::PointCloud<pcl::PointXYZ>::Ptr PCL_3D::transformToReferencePoint(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const Eigen::Vector3f& referencePoint)
{
    //Convert the reference point to pcl::PointXYZ
    pcl::PointXYZ referencePointXYZ(referencePoint.x(), referencePoint.y(), referencePoint.z());
    return segmentation->transformCloud(cloud, referencePointXYZ);
}


