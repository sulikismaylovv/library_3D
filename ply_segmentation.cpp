#include "ply_segmentation.h"
#include "Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h"
#include <pcl/common/transforms.h> // Make sure this include is present
#include <pcl/filters/passthrough.h>

#include "ply_processor.h"

std::vector<pcl::PointIndices> ply_segmentation::segmentAndExtractClusters(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    // Create the segmentation object for the planar model and set all the parameters
    SACSegmentation<PointXYZ> seg;
    PointIndices::Ptr inliers(new PointIndices);
    ModelCoefficients::Ptr coefficients(new ModelCoefficients);
    PointCloud<PointXYZ>::Ptr cloud_plane(new PointCloud<PointXYZ>());
    PCDWriter writer;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(SACMODEL_PLANE);
    seg.setMethodType(SAC_RANSAC);
    // Explanation of the parameters:
    // Distance Threshold - the maximum distance a point can be from the model and still be considered an inlier (in meters)
    // MaxIterations - the maximum number of iterations the algorithm will run for
    seg.setMaxIterations(1000);
    seg.setDistanceThreshold(0.5);

    // Assume a maximum number of iterations or stop if the change in removed points is small
    int max_iterations = 10;
    int iterations = 0;
    int previous_size = static_cast<int>(cloud->size());
    int delta_threshold = 25; // Minimum change in size to continue segmentation

    while (iterations < max_iterations) {
        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);
        if (inliers->indices.size() == 0 || (previous_size - static_cast<int>(cloud->size()) < delta_threshold)) {
            std::cout << "Could not estimate a planar model for the given dataset or change in point cloud size is too small." << std::endl;
            break;
        }

        // Check the normal of the plane if necessary
        Eigen::Vector3f plane_normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
        // For instance, if you expect the normal to be roughly pointing up, you can check if the z-component is positive
        // Adjust the tolerance according to your scenario
        if (std::abs(plane_normal.dot(Eigen::Vector3f::UnitZ())) < 0.9) {
            std::cout << "Plane normal is not pointing upwards." << std::endl;
            break; // This is not the plane we are looking for, so we stop the process
        }

        // Extract the planar inliers from the input cloud
        ExtractIndices<PointXYZ> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*cloud_plane); // Extracted but not used

        // Remove the planar inliers, extract the rest
        extract.setNegative(true);
        PointCloud<PointXYZ>::Ptr cloud_f(new PointCloud<PointXYZ>());
        extract.filter(*cloud_f);

        // Update cloud and sizes for the next iteration
        *cloud = *cloud_f;
        previous_size = static_cast<int>(cloud->size());

        // Increment the iteration counter
        iterations++;
    }

    // Creating the KdTree object for the search method of the extraction
    search::KdTree<PointXYZ>::Ptr tree(new search::KdTree<PointXYZ>());
    tree->setInputCloud(cloud);

    std::vector<PointIndices> cluster_indices;
    EuclideanClusterExtraction<PointXYZ> ec;
    // Explanation of the parameters:
    // Cluster Tolerance - the maximum distance between points that belong to the same cluster
    // Min Cluster Size - the minimum number of points that a cluster needs to contain in order to be considered valid
    // Max Cluster Size - the maximum number of points that a cluster needs to contain in order to be considered valid (useful for filtering noise)
    // For example , cluster tolerance of 11 means 11mm
    ec.setClusterTolerance(13.0);
    ec.setMinClusterSize(500);
    ec.setMaxClusterSize(25000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    return cluster_indices;
}

std::vector<pcl::PointIndices> ply_segmentation::segmentAndExtractForCalibration(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    // Create the segmentation object for the planar model and set all the parameters
    SACSegmentation<PointXYZ> seg;
    PointIndices::Ptr inliers(new PointIndices);
    ModelCoefficients::Ptr coefficients(new ModelCoefficients);
    PointCloud<PointXYZ>::Ptr cloud_plane(new PointCloud<PointXYZ>());
    PCDWriter writer;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(SACMODEL_PLANE);
    seg.setMethodType(SAC_RANSAC);
    // Explanation of the parameters:
    // Distance Threshold - the maximum distance a point can be from the model and still be considered an inlier (in meters)
    // MaxIterations - the maximum number of iterations the algorithm will run for
    seg.setMaxIterations(1000);
    seg.setDistanceThreshold(0.5);

    int nr_points = static_cast<int>(cloud->size());
    while (cloud->size() > 0.3 * nr_points) {
        // Segment the largest planar component from the remaining cloud
        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);
        if (inliers->indices.size() == 0) {
            std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
            break;
        }

        // Extract the planar inliers from the input cloud
        ExtractIndices<PointXYZ> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(false);

        // Get the points associated with the planar surface
        extract.filter(*cloud_plane);
        //std::cout << "PointCloud representing the planar component: " << cloud_plane->size() << " data points." << std::endl;

        // Remove the planar inliers, extract the rest
        extract.setNegative(true);
        PointCloud<PointXYZ>::Ptr cloud_f(new PointCloud<PointXYZ>());
        extract.filter(*cloud_f);
        *cloud = *cloud_f;
    }

    // Creating the KdTree object for the search method of the extraction
    search::KdTree<PointXYZ>::Ptr tree(new search::KdTree<PointXYZ>());
    tree->setInputCloud(cloud);

    std::vector<PointIndices> cluster_indices;
    EuclideanClusterExtraction<PointXYZ> ec;
    // Explanation of the parameters:
    // Cluster Tolerance - the maximum distance between points that belong to the same cluster
    // Min Cluster Size - the minimum number of points that a cluster needs to contain in order to be considered valid
    // Max Cluster Size - the maximum number of points that a cluster needs to contain in order to be considered valid (useful for filtering noise)
    // For example , cluster tolerance of 11 means 11mm
    ec.setClusterTolerance(15.0);
    ec.setMinClusterSize(500);
    ec.setMaxClusterSize(15000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    return cluster_indices;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr ply_segmentation::extractLargestCluster(const std::vector<pcl::PointIndices>& cluster_indices, const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    if (cluster_indices.empty()) {
        PCL_ERROR("No clusters found.\n");
        return nullptr;
    }

    // Assume the first cluster is the largest for initialization
    size_t largest_cluster_idx = 0;
    size_t largest_size = 0;
    int cluster_id = 0;

    // Find the largest cluster
    for (const auto& cluster : cluster_indices) {
        if (cluster.indices.size() > largest_size) {
            largest_size = cluster.indices.size();
            largest_cluster_idx = cluster_id;
        }
        cluster_id++;
    }

    // Extract the largest cluster
    PointCloud<PointXYZ>::Ptr largest_cluster(new PointCloud<PointXYZ>());
    for (const auto& idx :  cluster_indices[largest_cluster_idx].indices) {
        largest_cluster->push_back((*cloud)[idx]);
    }

    return largest_cluster;
}

void ply_segmentation::visualizePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::vector<pcl::PointIndices>& cluster_indices) {
    // Initialize viewer
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Cluster viewer"));
    viewer->setBackgroundColor(0, 0, 0);

    // Add a coordinate system to the viewer
    viewer->addCoordinateSystem(50);

    // Add the point cloud
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> originalColor(cloud, 255, 255, 255);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, originalColor, "cloud");

    // Random number generation for colors
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    // Highlight each cluster with a bounding box
    for (int cluster_id = 0; cluster_id < cluster_indices.size(); ++cluster_id) {
        const auto& indices = cluster_indices[cluster_id].indices;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        for (int idx : indices) {
            cluster_cloud->push_back((*cloud)[idx]);
        }

        // Generate a random color for the cluster
        int r = dis(gen);
        int g = dis(gen);
        int b = dis(gen);

        // Get the bounding box points
        pcl::PointXYZ minPt, maxPt;
        pcl::getMinMax3D(*cluster_cloud, minPt, maxPt);
        minPt.z = 0; // Set min Z to the origin level for visualization


        // Define the 8 vertices of the bounding box
        pcl::PointXYZ p1(minPt.x, minPt.y, minPt.z);
        pcl::PointXYZ p2(minPt.x, maxPt.y, minPt.z);
        pcl::PointXYZ p3(maxPt.x, maxPt.y, minPt.z);
        pcl::PointXYZ p4(maxPt.x, minPt.y, minPt.z);
        pcl::PointXYZ p5(minPt.x, minPt.y, maxPt.z);
        pcl::PointXYZ p6(minPt.x, maxPt.y, maxPt.z);
        pcl::PointXYZ p7(maxPt.x, maxPt.y, maxPt.z);
        pcl::PointXYZ p8(maxPt.x, minPt.y, maxPt.z);

        // Draw lines between the corners (edges) of the bounding box
        viewer->addLine(p1, p2, r / 255.0, g / 255.0, b / 255.0, "line1_" + std::to_string(cluster_id));
        viewer->addLine(p2, p3, r / 255.0, g / 255.0, b / 255.0, "line2_" + std::to_string(cluster_id));
        viewer->addLine(p3, p4, r / 255.0, g / 255.0, b / 255.0, "line3_" + std::to_string(cluster_id));
        viewer->addLine(p4, p1, r / 255.0, g / 255.0, b / 255.0, "line4_" + std::to_string(cluster_id));
        viewer->addLine(p5, p6, r / 255.0, g / 255.0, b / 255.0, "line5_" + std::to_string(cluster_id));
        viewer->addLine(p6, p7, r / 255.0, g / 255.0, b / 255.0, "line6_" + std::to_string(cluster_id));
        viewer->addLine(p7, p8, r / 255.0, g / 255.0, b / 255.0, "line7_" + std::to_string(cluster_id));
        viewer->addLine(p8, p5, r / 255.0, g / 255.0, b / 255.0, "line8_" + std::to_string(cluster_id));
        viewer->addLine(p1, p5, r / 255.0, g / 255.0, b / 255.0, "line9_" + std::to_string(cluster_id));
        viewer->addLine(p2, p6, r / 255.0, g / 255.0, b / 255.0, "line10_" + std::to_string(cluster_id));
        viewer->addLine(p3, p7, r / 255.0, g / 255.0, b / 255.0, "line11_" + std::to_string(cluster_id));
        viewer->addLine(p4, p8, r / 255.0, g / 255.0, b / 255.0, "line12_" + std::to_string(cluster_id));

        // Print dimensions of bounding box
        std::cout << "Bounding Box " << cluster_id << " dimensions: "
                  << maxPt.x - minPt.x << " "
                  << maxPt.y - minPt.y << " "
                  << maxPt.z - minPt.z << std::endl;
    }

    // Spin until 'q' is pressed
    while (!viewer->wasStopped()) {
        viewer->spinOnce(10000);
    }
}
pcl::PointCloud<pcl::PointXYZ>::Ptr ply_segmentation::extractCluster(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const pcl::PointIndices& cluster)
{
    PointCloud<PointXYZ>::Ptr cluster_cloud(new PointCloud<PointXYZ>());
    for (const auto& idx : cluster.indices) {
        cluster_cloud->push_back((*cloud)[idx]);
    }

    return cluster_cloud;
}

// transform cloud
PointCloud<PointXYZ>::Ptr ply_segmentation::transformCloud(const PointCloud<PointXYZ>::Ptr& cloud, const PointXYZ& reference_point) {
    //transofrm existing cloud to the reference point
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << -reference_point.x, -reference_point.y, -reference_point.z;
    transformPointCloud(*cloud, *cloud, transform);
    return cloud;
}


std::vector<ClusterInfo> ply_segmentation::extractLocations(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::vector<pcl::PointIndices>& cluster_indices) {
    std::vector<ClusterInfo> clusters;

    for (const auto& cluster : cluster_indices) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        for (const auto& idx : cluster.indices) {
            cluster_cloud->push_back((*cloud)[idx]);
        }

        //print cluster size
        std::cout << "Cluster size: " << cluster_cloud->size() << std::endl;

        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cluster_cloud, centroid);
        Eigen::Matrix3f covariance;
        pcl::computeCovarianceMatrixNormalized(*cluster_cloud, centroid, covariance);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
        Eigen::Matrix3f eigen_vectors = eigen_solver.eigenvectors();

        // Find the principal component (largest eigenvector)
        int principal_component_idx = std::distance(eigen_solver.eigenvalues().data(), std::max_element(eigen_solver.eigenvalues().data(), eigen_solver.eigenvalues().data() + 3));

        // Create a rotation matrix to align this principal component with the X-axis
        Eigen::Matrix3f rotation_matrix;
        rotation_matrix = Eigen::AngleAxisf(atan2(eigen_vectors.col(principal_component_idx)(1), eigen_vectors.col(principal_component_idx)(0)), Eigen::Vector3f::UnitZ());

        // Rotate the point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr rotated_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        // Now transform the entire cloud
        pcl::transformPointCloud(*cluster_cloud, *rotated_cloud, Eigen::Affine3f(rotation_matrix));

        // Recompute the bounding box on the rotated cloud
        pcl::PointXYZ minPt, maxPt;
        pcl::getMinMax3D(*rotated_cloud, minPt, maxPt);
        minPt.z = 10; // Set min Z to the origin level for visualization

        ClusterInfo info;
        info.centroid = centroid;
        info.dimensions = Eigen::Vector3f(maxPt.x - minPt.x, maxPt.y - minPt.y, maxPt.z - minPt.z);
        Eigen::Quaternionf quat(rotation_matrix);
        info.orientation = quat; // Store the orientation
        info.clusterId = clusters.size() + 1;
        info.eulerAngles = quat.toRotationMatrix().eulerAngles(2, 1, 0);
        info.clusterSize = cluster_cloud->size();

        clusters.push_back(info);


        // Enhanced visualization to check alignments
        //std::unique_ptr<ply_processor> processor = std::make_unique<ply_processor>();
        //processor->visualizePointCloud(rotated_cloud);
        //processor->visualizePointCloudV2(cluster_cloud, rotated_cloud, centroid, eigen_vectors);
    }

    return clusters;
}








//find reference point
pcl::PointXYZ ply_segmentation::findReferencePoint(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    // Assuming cloud is not empty
    if (cloud->points.empty()) {
        throw std::runtime_error("The point cloud is empty.");
    }

    pcl::PointXYZ minPt, maxPt;
    pcl::getMinMax3D(*cloud, minPt, maxPt);

    // Find the point with maximum X and minimum Y, assuming Z is height and can be disregarded for bottom right calculation
    pcl::PointXYZ reference_point;
    reference_point.x = maxPt.x;
    reference_point.y = maxPt.y;
    reference_point.z = minPt.z;

    return reference_point;
}

//function to subtract point clouds
PointCloud<PointXYZ>::Ptr ply_segmentation::subtractPointClouds(const PointCloud<PointXYZ>::Ptr& cloudA, const PointCloud<PointXYZ>::Ptr& cloudB, float searchRadius) {
    // KdTree for searching
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloudB); // Use cloudB for searching

    // Indices to keep
    pcl::PointIndices::Ptr keepIndices(new pcl::PointIndices());

    // Search for each point in cloudA within the radius in cloudB
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    for (size_t i = 0; i < cloudA->points.size(); ++i) {
        if (kdtree.radiusSearch(cloudA->points[i], searchRadius, pointIdxRadiusSearch, pointRadiusSquaredDistance) == 0) {
            // If no points are found within the radius, add the index to keepIndices
            keepIndices->indices.push_back(i);
        }
    }

    // Extract the points based on the indices
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    pcl::PointCloud<pcl::PointXYZ>::Ptr result(new pcl::PointCloud<pcl::PointXYZ>);
    extract.setInputCloud(cloudA);
    extract.setIndices(keepIndices);
    extract.setNegative(false); // False to keep the points in keepIndices
    extract.filter(*result);

    return result;
}

//Function to extract location from cloud
std::vector<ClusterInfo> ply_segmentation::extractLocationsCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    std::vector<ClusterInfo> clusters;

    //print cluster size
    std::cout << "Cluster size: " << cloud->size() << std::endl;

    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);
    Eigen::Matrix3f covariance;
    pcl::computeCovarianceMatrixNormalized(*cloud, centroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigen_vectors = eigen_solver.eigenvectors();

    // Find the principal component (largest eigenvector)
    int principal_component_idx = std::distance(eigen_solver.eigenvalues().data(), std::max_element(eigen_solver.eigenvalues().data(), eigen_solver.eigenvalues().data() + 3));

    // Create a rotation matrix to align this principal component with the X-axis
    Eigen::Matrix3f rotation_matrix;
    rotation_matrix = Eigen::AngleAxisf(atan2(eigen_vectors.col(principal_component_idx)(1), eigen_vectors.col(principal_component_idx)(0)), Eigen::Vector3f::UnitZ());

    // Rotate the point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr rotated_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    // Now transform the entire cloud
    pcl::transformPointCloud(*cloud, *rotated_cloud, Eigen::Affine3f(rotation_matrix));

    // Recompute the bounding box on the rotated cloud
    pcl::PointXYZ minPt, maxPt;
    pcl::getMinMax3D(*rotated_cloud, minPt, maxPt);
    minPt.z = 10; // Set min Z to the origin level for visualization

    ClusterInfo info;
    info.centroid = centroid;
    info.dimensions = Eigen::Vector3f(maxPt.x - minPt.x, maxPt.y - minPt.y, maxPt.z - minPt.z);
    Eigen::Quaternionf quat(rotation_matrix);
    info.orientation = quat; // Store the orientation
    info.clusterId = clusters.size() + 1;
    info.eulerAngles = quat.toRotationMatrix().eulerAngles(2, 1, 0);
    info.clusterSize = cloud->size();

    clusters.push_back(info);

    //std::unique_ptr<ply_processor> processor = std::make_unique<ply_processor>();
    //processor->visualizePointCloud(rotated_cloud);


    return clusters;

}
