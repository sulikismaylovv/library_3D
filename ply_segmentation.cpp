#include "ply_segmentation.h"
#include "Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h"
#include <pcl/common/transforms.h> // Make sure this include is present
#include <pcl/filters/passthrough.h>

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
    ec.setClusterTolerance(15);
    ec.setMinClusterSize(1000);
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

void ply_segmentation::visualizePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::vector<pcl::PointIndices>& cluster_indices)
{
    // Initialize viewer
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Cluster viewer"));
    viewer->setBackgroundColor(0, 0, 0);

    // Add arrows to the viewer of x, y, z from origin of length 100
    viewer->addCoordinateSystem(50);

    // Add the point cloud
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> originalColor(cloud, 255, 255, 255); // White color
    viewer->addPointCloud<pcl::PointXYZ>(cloud, originalColor, "cloud");
    // Setup random number generation for colors
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    // Highlight each cluster in the cloud with a unique color and add a bounding box
    int cluster_id = 0;
    for (const auto& cluster : cluster_indices) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        for (const auto& idx : cluster.indices) {
            cluster_cloud->push_back((*cloud)[idx]);
        }

        // Generate a random color for the cluster
        int r = dis(gen);
        int g = dis(gen);
        int b = dis(gen);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> clusterColor(cluster_cloud, r, g, b);
        viewer->addPointCloud<pcl::PointXYZ>(cluster_cloud, clusterColor, "cluster" + std::to_string(cluster_id));

        // Calculate and add a bounding box for each cluster with minPt.z set to 0
        pcl::PointXYZ minPt, maxPt;
        pcl::getMinMax3D(*cluster_cloud, minPt, maxPt);
        minPt.z = 0; // Set min Z to the origin level for visualization

        // Add bounding box with random color
        viewer->addCube(minPt.x, maxPt.x, minPt.y, maxPt.y, minPt.z, minPt.z, r / 255.0, g / 255.0, b / 255.0, "bbox" + std::to_string(cluster_id));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, "bbox" + std::to_string(cluster_id));

        //print dimensions of bound box:
        std::cout << "Bounding Box " << cluster_id << " dimensions: " << maxPt.x - minPt.x << " " << maxPt.y - minPt.y << " " << maxPt.z - minPt.z << std::endl;


        cluster_id++;
    }

    // Spin until 'q' is pressed
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
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

//extract locations
std::vector<ClusterInfo> ply_segmentation::extractLocations(const PointCloud<PointXYZ>::Ptr& cloud, const std::vector<pcl::PointIndices>& cluster_indices) {
    std::vector<ClusterInfo> clusters;

    for (const auto& cluster : cluster_indices) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        for (const auto& idx : cluster.indices) {
            cluster_cloud->push_back((*cloud)[idx]);
        }

        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cluster_cloud, centroid);
        pcl::PointXYZ minPt, maxPt;
        pcl::getMinMax3D(*cluster_cloud, minPt, maxPt);
        minPt.z = 0; // Set min Z to the origin level for visualization

        Eigen::Matrix3f covariance;
        pcl::computeCovarianceMatrixNormalized(*cluster_cloud, centroid, covariance);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
        Eigen::Matrix3f eigen_vectors = eigen_solver.eigenvectors();
        Eigen::Quaternionf quat(eigen_vectors);

        ClusterInfo info;
        info.centroid = centroid;
        info.dimensions = Eigen::Vector3f(maxPt.x - minPt.x, maxPt.y - minPt.y, maxPt.z - minPt.z);
        info.orientation = quat; // Use the computed quaternion
        info.clusterId = clusters.size() + 1; // Or any other identifier

        clusters.push_back(info);
    }

    return clusters;
}

//find reference point
PointXYZ ply_segmentation::findReferencePoint(const PointCloud<PointXYZ>::Ptr& cloud) {
    //reference point is at bottom left corner of the cloud
    pcl::PointXYZ minPt, maxPt;
    pcl::getMinMax3D(*cloud, minPt, maxPt);
    //adjust height
    //minPt.z += 20;
    // The reference point is the minimum point (bottom left)
    pcl::PointXYZ reference_point = minPt;
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

