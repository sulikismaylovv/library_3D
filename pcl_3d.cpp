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
    auto cloud = cloudOpt.value(); // Dereference std::optionalSS

    // Step 1: Apply Z PassThrough Filter
    if (!processor->applyPassThroughZOnly(*cloud, height, height+20)) {
        std::cerr << "Applying PassThrough Z filter failed." << std::endl;
        throw std::runtime_error("Applying PassThrough Z filter failed.");
    }

    // Step 2: Apply Voxel Grid Filter
    if (!processor->applyVoxelGridFilter(*cloud, 0.5f)) { // Example leaf size
        std::cerr << "Applying Voxel Grid filter failed." << std::endl;
        throw std::runtime_error("Applying Voxel Grid filter failed.");
    }

    // Step 3: Apply MLS Surface Reconstruction
    if (!processor->applyMLSSurfaceReconstruction(*cloud, 1.5f)) { // Example search radius
        std::cerr << "Applying MLS Surface Reconstruction failed." << std::endl;
        throw std::runtime_error("Applying MLS Surface Reconstruction failed.");
    }

    // Step 4: Remove Outliers
    if (!processor->removeOutliers(*cloud, 5, 1.7)) { // Example meanK and stddevMulThresh
        std::cerr << "Removing outliers failed." << std::endl;
        throw std::runtime_error("Removing outliers failed.");
    }

    // Step 5: Segment and Extract Clusters
    auto cluster_indices = segmentation->segmentAndExtractForCalibration(cloud);

    // Extract the largest cluster and find the reference point
    auto largest_cluster = segmentation->extractLargestCluster(cluster_indices, cloud);
    pcl::PointXYZ reference_point = segmentation->findReferencePoint(largest_cluster);

    reference_point.z += 10;

    std::cout << "Reference point: " << reference_point.x << " " << reference_point.y << " " << reference_point.z << std::endl;

    //transform the point cloud to the reference point
    segmentation->transformCloud(cloud, reference_point);

    // Step 6: Invert the point cloud
    if (!processor->invertPointCloud(*cloud)) {
        std::cerr << "Inverting point cloud failed." << std::endl;
        throw std::runtime_error("Inverting point cloud failed.");
    }

    //Convert the reference point to Eigen::Vector3f
    Eigen::Vector3f reference_point_eigen(reference_point.x, reference_point.y, reference_point.z);

    std::cout << "Reference point (Eigen): " << reference_point_eigen.transpose() << std::endl;

    return reference_point_eigen;

}

//function to find objects
std::vector<ClusterInfo> PCL_3D::findBoundingBox(const std::string& filePathBox,
                                                 const std::string& filePathTray,
                                                 const Eigen::Vector3f& referencePoint,
                                                 const Eigen::Vector3f& prevLocation,
                                                 const Eigen::Vector3f& dimensions){
    // Compute the absolute location from the previous location and reference point
    Eigen::Vector3f absPrevLocation = referencePoint - prevLocation;

    // Determine the bounding box around the previous location in the point cloud's original coordinates
    pcl::PointXYZ minPt, maxPt;

    //Half the dimensions
    Eigen::Vector3f halfDimensions = dimensions/2.0f;

    //Initiate info
    std::vector<ClusterInfo> info;

    std::cout << "Half Dimensions: " << halfDimensions.x() << " " << halfDimensions.y() << " " << halfDimensions.z() << std::endl;
    std::cout << "Prev Location: " << prevLocation.x() << " " << prevLocation.y() << " " << prevLocation.z() << std::endl;
    std::cout << "Abs Prev Location: " << absPrevLocation.x() << " " << absPrevLocation.y() << " " << absPrevLocation.z() << std::endl;

    if(prevLocation.isZero()) {
        // Use reference point if previous location is zero (i.e., uninitialized)
        minPt = {referencePoint.x() - 1000, referencePoint.y() - 1000, referencePoint.z() - 2500};
        maxPt = {referencePoint.x(), referencePoint.y(), referencePoint.z()};
    } else {
        if(dimensions.isZero()){
            // Use default dimensions if not provided
            minPt = {absPrevLocation.x() - 300, absPrevLocation.y() - 300, referencePoint.z() - 2500};
            maxPt = {absPrevLocation.x() + 300, absPrevLocation.y() + 300, referencePoint.z()};
        }else{
            minPt = {absPrevLocation.x() - halfDimensions.x(), absPrevLocation.y() - halfDimensions.y(), referencePoint.z() - 2500};
            maxPt = {absPrevLocation.x() + halfDimensions.x(), absPrevLocation.y() + halfDimensions.y(), referencePoint.z()};
        }

    }


    //print min and max values
    std::cout << "Min values: " << minPt.x << " " << minPt.y << " " << minPt.z << std::endl;
    std::cout << "Max values: " << maxPt.x << " " << maxPt.y << " " << maxPt.z << std::endl;

    // Step 1: Load the point cloud with the object
    auto cloudOpt = processor->loadCloud(filePathBox);
    if (!cloudOpt) {
        std::cerr << "Failed to read point cloud." << std::endl;
        throw std::runtime_error("Failed to read point cloud.");
    }

    auto cloud = cloudOpt.value(); // Dereference std::optional

    // Step 2: Apply PassThrough Filter
    if (!processor->applyPassthroughFilter(*cloud, minPt, maxPt)) {
        std::cerr << "Applying PassThrough filter failed." << std::endl;
        throw std::runtime_error("Applying PassThrough filter failed.");
    }

    //processor->visualizePointCloud(cloud);

    if(prevLocation.isZero()){
        //Step 3: Load the point cloud with the tray
        auto cloudTrayOpt = processor->loadCloud(filePathTray);
        if (!cloudTrayOpt) {
            std::cerr << "Failed to read point cloud." << std::endl;
            throw std::runtime_error("Failed to read point cloud.");
        }

        auto cloudTray = cloudTrayOpt.value(); // Dereference std::optional

        // Step 4: Apply PassThrough Filter
        if (!processor->applyPassthroughFilter(*cloudTray, minPt, maxPt)) {
            std::cerr << "Applying PassThrough filter failed." << std::endl;
            throw std::runtime_error("Applying PassThrough filter failed.");
        }


        // Step 5: Subtract the tray from the object
        auto isolated_pcl = segmentation->subtractPointClouds(cloud, cloudTray, 40.0f);
        if (isolated_pcl->size() == 0) {
            std::cerr << "Subtracting point clouds failed.(no boxes)" << std::endl;
            return {};
        }

        // //Optional, statistical outlier removal
        // if (!processor->removeOutliers(*isolated_pcl, 5, 1.7)) { // Example meanK and stddevMulThresh
        //     std::cerr << "Removing outliers failed." << std::endl;
        //     throw std::runtime_error("Removing outliers failed.");
        // }

        // Step 6: Segment and Extract Clusters
        auto cluster_indices = segmentation->segmentAndExtractClusters(isolated_pcl);

        // // Additional step perform outlier removal on clusers
        // for (auto& cluster : cluster_indices) {
        //     auto cluster_cloud = segmentation->extractCluster(isolated_pcl, cluster);
        //     if (!processor->removeOutliers(*cluster_cloud, 0.05f, 150)) { // Example meanK and stddevMulThresh
        //         std::cerr << "Removing outliers failed." << std::endl;
        //         throw std::runtime_error("Removing outliers failed.");
        //     }
        // }

        // Step 7: Transform the point cloud to align with the specified reference point
        pcl::PointXYZ referencePointXYZ(referencePoint.x(), referencePoint.y(), referencePoint.z());
        auto transformed_cloud = segmentation->transformCloud(isolated_pcl, referencePointXYZ);

        //Step 8:Invert the point cloud
        if (!processor->invertPointCloud(*transformed_cloud)) {
            std::cerr << "Inverting point cloud failed." << std::endl;
            throw std::runtime_error("Inverting point cloud failed.");
        }

        //Step 9: Extract locations
        info = segmentation->extractLocations(transformed_cloud, cluster_indices);


        // Step 10: Visualize the results
        //segmentation->visualizePointCloud(transformed_cloud, cluster_indices);
    }
    else{
        //processor->visualizePointCloud(cloud);
        //Step 3: Load the point cloud with the tray
        auto cloudTrayOpt = processor->loadCloud(filePathTray);
        if (!cloudTrayOpt) {
            std::cerr << "Failed to read point cloud." << std::endl;
            throw std::runtime_error("Failed to read point cloud.");
        }

        auto cloudTray = cloudTrayOpt.value(); // Dereference std::optional

        // Step 4: Apply PassThrough Filter
        if (!processor->applyPassthroughFilter(*cloudTray, minPt, maxPt)) {
            std::cerr << "Applying PassThrough filter failed." << std::endl;
            throw std::runtime_error("Applying PassThrough filter failed.");
        }


        // Step 5: Subtract the tray from the object
        auto isolated_pcl = segmentation->subtractPointClouds(cloud, cloudTray, 40.0f);
        if (isolated_pcl->size() == 0) {
            std::cerr << "Subtracting point clouds failed.(no boxes)" << std::endl;
            return {};
        }

        //Convert the reference point to pcl::PointXYZ
        pcl::PointXYZ referencePointXYZ(referencePoint.x(), referencePoint.y(), referencePoint.z());
        auto transformed_cloud = segmentation->transformCloud(isolated_pcl, referencePointXYZ);

        //Step 8:Invert the point cloud
        if (!processor->invertPointCloud(*transformed_cloud)) {
            std::cerr << "Inverting point cloud failed." << std::endl;
            throw std::runtime_error("Inverting point cloud failed.");
        }

        //Extract Location from One cluster
        info = segmentation->extractLocationsCloud(transformed_cloud);

        //processor->visualizePointCloud(transformed_cloud);
    }

    return info;

}

//function to transform point cloud to reference point
pcl::PointCloud<pcl::PointXYZ>::Ptr PCL_3D::transformToReferencePoint(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const Eigen::Vector3f& referencePoint)
{
    //Convert the reference point to pcl::PointXYZ
    pcl::PointXYZ referencePointXYZ(referencePoint.x(), referencePoint.y(), referencePoint.z());

    //inver pcl
    processor->invertPointCloud(*cloud);

    return segmentation->transformCloud(cloud, referencePointXYZ);
}

//function to preprocess point cloud
pcl::PointCloud<pcl::PointXYZ>::Ptr PCL_3D::preprocessPointCloud(const std::string& filePathBox,
                                                                 const std::string &filePathTray,
                                                                 const Eigen::Vector3f& referencePoint,
                                                                 const Eigen::Vector3f& prevLocation,
                                                                 const Eigen::Vector3f& dimensions)
{
    // Compute the absolute location from the previous location and reference point
    Eigen::Vector3f absPrevLocation = referencePoint - prevLocation;

    // Determine the bounding box around the previous location in the point cloud's original coordinates
    pcl::PointXYZ minPt, maxPt;

    //Half the dimensions
    Eigen::Vector3f halfDimensions = dimensions/2.0f;

    //Initiate info

    std::cout << "Half Dimensions: " << halfDimensions.x() << " " << halfDimensions.y() << " " << halfDimensions.z() << std::endl;
    std::cout << "Prev Location: " << prevLocation.x() << " " << prevLocation.y() << " " << prevLocation.z() << std::endl;
    std::cout << "Abs Prev Location: " << absPrevLocation.x() << " " << absPrevLocation.y() << " " << absPrevLocation.z() << std::endl;

    if(prevLocation.isZero()) {
        // Use reference point if previous location is zero (i.e., uninitialized)
        minPt = {referencePoint.x() - 1000, referencePoint.y() - 1000, referencePoint.z() - 2500};
        maxPt = {referencePoint.x(), referencePoint.y(), referencePoint.z()};
    } else {
        if(dimensions.isZero()){
            // Use default dimensions if not provided
            minPt = {absPrevLocation.x() - 300, absPrevLocation.y() - 300, referencePoint.z() - 2500};
            maxPt = {absPrevLocation.x() + 300, absPrevLocation.y() + 300, referencePoint.z()};
        }else{
            minPt = {absPrevLocation.x() - halfDimensions.x(), absPrevLocation.y() - halfDimensions.y(), referencePoint.z() - 2500};
            maxPt = {absPrevLocation.x() + halfDimensions.x(), absPrevLocation.y() + halfDimensions.y(), referencePoint.z()};
        }

    }

    //print min and max values
    std::cout << "Min values: " << minPt.x << " " << minPt.y << " " << minPt.z << std::endl;
    std::cout << "Max values: " << maxPt.x << " " << maxPt.y << " " << maxPt.z << std::endl;

    // Step 1: Load the point cloud with the object
    auto cloudOpt = processor->loadCloud(filePathBox);
    if (!cloudOpt) {
        std::cerr << "Failed to read point cloud." << std::endl;
        throw std::runtime_error("Failed to read point cloud.");
    }

    auto cloud = cloudOpt.value(); // Dereference std::optional

    // Step 2: Apply PassThrough Filter
    if (!processor->applyPassthroughFilter(*cloud, minPt, maxPt)) {
        std::cerr << "Applying PassThrough filter failed." << std::endl;
        throw std::runtime_error("Applying PassThrough filter failed.");
    }


    if(prevLocation.isZero()){
        //Step 3: Load the point cloud with the tray
        auto cloudTrayOpt = processor->loadCloud(filePathTray);
        if (!cloudTrayOpt) {
            std::cerr << "Failed to read point cloud." << std::endl;
            throw std::runtime_error("Failed to read point cloud.");
        }

        auto cloudTray = cloudTrayOpt.value(); // Dereference std::optional

        // Step 4: Apply PassThrough Filter
        if (!processor->applyPassthroughFilter(*cloudTray, minPt, maxPt)) {
            std::cerr << "Applying PassThrough filter failed." << std::endl;
            throw std::runtime_error("Applying PassThrough filter failed.");
        }

        // Step 5: Subtract the tray from the object
        auto isolated_pcl = segmentation->subtractPointClouds(cloud, cloudTray, 40.0f);
        if (isolated_pcl->size() == 0) {
            std::cerr << "Subtracting point clouds failed.(no boxes)" << std::endl;
            return {};
        }

        //Optional, statistical outlier removal
        // if (!processor->removeOutliers(*isolated_pcl, 5, 1.7)) { // Example meanK and stddevMulThresh
        //     std::cerr << "Removing outliers failed." << std::endl;
        //     throw std::runtime_error("Removing outliers failed.");
        // }

        return isolated_pcl;
    }
    else{
        return cloud;
    }

}


//Light Function to find bounding box
std::vector<ClusterInfo>  PCL_3D::findBoundingBoxLight(const pcl::PointCloud<pcl::PointXYZ>::Ptr& isolated_pcl,
                                                      const Eigen::Vector3f& referencePoint,
                                                      const Eigen::Vector3f& prevLocation){
    //Initiate info
    std::vector<ClusterInfo> info;

    if(prevLocation.isZero()) {
        // Step 6: Segment and Extract Clusters
        auto cluster_indices = segmentation->segmentAndExtractClusters(isolated_pcl);

        // Additional step perform outlier removal on clusers
        for (auto& cluster : cluster_indices) {
            auto cluster_cloud = segmentation->extractCluster(isolated_pcl, cluster);
            if (!processor->removeOutliers(*cluster_cloud, 0.05f, 150)) { // Example meanK and stddevMulThresh
                std::cerr << "Removing outliers failed." << std::endl;
                throw std::runtime_error("Removing outliers failed.");
            }
        }

        // Step 7: Transform the point cloud to align with the specified reference point
        //Convert the reference point to pcl::PointXYZ
        pcl::PointXYZ referencePointXYZ(referencePoint.x(), referencePoint.y(), referencePoint.z());
        auto transformed_cloud = segmentation->transformCloud(isolated_pcl, referencePointXYZ);

        //Step 8:Invert the point cloud
        if (!processor->invertPointCloud(*transformed_cloud)) {
            std::cerr << "Inverting point cloud failed." << std::endl;
            throw std::runtime_error("Inverting point cloud failed.");
        }

        //Step 9: Extract locations
        auto info = segmentation->extractLocations(transformed_cloud, cluster_indices);

        // Step 10: Visualize the results
        segmentation->visualizePointCloud(transformed_cloud, cluster_indices);}
    else{
        //Convert the reference point to pcl::PointXYZ
        pcl::PointXYZ referencePointXYZ(referencePoint.x(), referencePoint.y(), referencePoint.z());
        auto transformed_cloud = segmentation->transformCloud(isolated_pcl, referencePointXYZ);

        //Step 8:Invert the point cloud
        if (!processor->invertPointCloud(*transformed_cloud)) {
            std::cerr << "Inverting point cloud failed." << std::endl;
            throw std::runtime_error("Inverting point cloud failed.");
        }

        //Extract Location from One cluster
        auto info = segmentation->extractLocationsCloud(transformed_cloud);

        //processor->visualizePointCloud(transformed_cloud);
    }


    return info;
}

