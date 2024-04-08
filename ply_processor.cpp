#include "ply_processor.h"
#include "pcl/filters/radius_outlier_removal.h"
#include <pcl/io/auto_io.h>
#include <pcl/filters/filter.h> // For removeNaNFromPointCloud

std::optional<pcl::PointCloud<pcl::PointXYZ>::Ptr> ply_processor::loadCloud(const std::string& filePath) {
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    if (pcl::io::load(filePath, *cloud) < 0) {
        PCL_ERROR("Failed to load cloud from %s.\n", filePath.c_str());
        return std::nullopt;
    }
    return cloud;
}

bool ply_processor::invertPointCloud(pcl::PointCloud<pcl::PointXYZ>& cloud) {
    if (cloud.empty()) return false;

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.matrix()(0, 0) = -1; //invert X axis
    transform.matrix()(1,1) = -1; // Invert Y axis
    transform.matrix()(2, 2) = -1; // Invert Z axis
    transformPointCloud(cloud, cloud, transform);
    return true;
}

bool ply_processor::applyVoxelGridFilter(pcl::PointCloud<pcl::PointXYZ>& cloud, float leafSize) {
    if (cloud.empty() || leafSize <= 0) return false;

    pcl::VoxelGrid<pcl::PointXYZ> filter;
    filter.setInputCloud(cloud.makeShared());
    filter.setLeafSize(leafSize, leafSize, leafSize);

    pcl::PointCloud<pcl::PointXYZ> filteredCloud;
    filter.filter(filteredCloud);
    cloud.swap(filteredCloud); // Use swap to minimize memory copy

    return true;
}

bool ply_processor::applyPassthroughFilter(pcl::PointCloud<pcl::PointXYZ>& cloud, const pcl::PointXYZ& minPt, const pcl::PointXYZ& maxPt) {
    if (cloud.empty()) return false;

    pcl::PassThrough<pcl::PointXYZ> filter;
    filter.setInputCloud(cloud.makeShared());

    // Apply filtering for each dimension
    filter.setFilterFieldName("x");
    filter.setFilterLimits(minPt.x, maxPt.x);
    pcl::PointCloud<pcl::PointXYZ> filteredCloud;
    filter.filter(filteredCloud);

    filter.setInputCloud(filteredCloud.makeShared());
    filter.setFilterFieldName("y");
    filter.setFilterLimits(minPt.y, maxPt.y);
    filter.filter(filteredCloud);

    filter.setInputCloud(filteredCloud.makeShared());
    filter.setFilterFieldName("z");
    filter.setFilterLimits(minPt.z, maxPt.z);
    filter.filter(filteredCloud);

    cloud.swap(filteredCloud); // Minimize memory copy
    return true;
}

bool ply_processor::applyPassThroughZOnly(pcl::PointCloud<pcl::PointXYZ>& cloud, float minZ, float maxZ) {
    if (cloud.empty() || minZ >= maxZ) return false;

    pcl::PassThrough<pcl::PointXYZ> filter;
    filter.setInputCloud(cloud.makeShared());
    filter.setFilterFieldName("z");
    filter.setFilterLimits(minZ, maxZ);

    pcl::PointCloud<pcl::PointXYZ> filteredCloud;
    filter.filter(filteredCloud);
    cloud.swap(filteredCloud); // Minimize memory copy

    return true;
}

bool ply_processor::applyMLSSurfaceReconstruction(pcl::PointCloud<pcl::PointXYZ>& cloud, float searchRadius) {
    if (cloud.empty() || searchRadius <= 0) return false;

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ> mlsPoints;

    pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> mls;
    mls.setInputCloud(cloud.makeShared());
    mls.setComputeNormals(true); // Set to true if you need normals
    mls.setPolynomialOrder(2); // 2nd order polynomial
    mls.setSearchMethod(tree);
    mls.setSearchRadius(searchRadius);
    mls.process(mlsPoints);

    cloud.swap(mlsPoints); // Use swap to minimize memory copy
    return true;
}

bool ply_processor::applyStatisticalOutlierRemoval(pcl::PointCloud<pcl::PointXYZ>& cloud, int meanK, double stddevMulThresh) {
    if (cloud.empty() || meanK <= 0 || stddevMulThresh <= 0) return false;

    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud.makeShared());
    sor.setMeanK(meanK);
    sor.setStddevMulThresh(stddevMulThresh);

    pcl::PointCloud<pcl::PointXYZ> filteredCloud;
    sor.filter(filteredCloud);

    cloud.swap(filteredCloud); // Minimize memory copy
    return true;
}

bool ply_processor::removeOutliers(pcl::PointCloud<pcl::PointXYZ>& cloud, float meanN, double radius) {
    if (cloud.empty() || meanN <= 0 || radius <= 0) return false;

    pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
    outrem.setInputCloud(cloud.makeShared());
    outrem.setRadiusSearch(radius);
    outrem.setMinNeighborsInRadius(meanN);
    outrem.filter(cloud);


    return true;
}

void ply_processor::visualizePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    // Downsample the cloud to improve efficiency, if necessary
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(0.01f, 0.01f, 0.01f); // Adjust the leaf size as necessary
    sor.filter(*cloud_filtered);

    // Create a viewer
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
    viewer->setBackgroundColor(0, 0, 0); // Black background
    viewer->addPointCloud<pcl::PointXYZ>(cloud_filtered, "cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

    // Add a coordinate system to the visualizer
    viewer->addCoordinateSystem(1.0);
    // Optionally, add labels to the axes
    viewer->addText3D("X", pcl::PointXYZ(1.1, 0, 0), 0.1, 1, 0, 0, "X_axis");
    viewer->addText3D("Y", pcl::PointXYZ(0, 1.1, 0), 0.1, 0, 1, 0, "Y_axis");
    viewer->addText3D("Z", pcl::PointXYZ(0, 0, 1.1), 0.1, 0, 0, 1, "Z_axis");

    viewer->initCameraParameters();

    // Process events
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100000); // Spin with a more reasonable delay
    }
}

