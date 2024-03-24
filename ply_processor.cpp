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

    for (auto& point : cloud) {
        point.y = -point.y;
        point.z = -point.z;
    }
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

bool ply_processor::removeOutliers(pcl::PointCloud<pcl::PointXYZ>& cloud, int meanN, double radius) {
    if (cloud.empty() || meanN <= 0 || radius <= 0) return false;

    pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
    outrem.setInputCloud(cloud.makeShared());
    outrem.setRadiusSearch(radius);
    outrem.setMinNeighborsInRadius(meanN);
    outrem.filter(cloud);


    return true;
}
