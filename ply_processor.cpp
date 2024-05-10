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
    viewer->addCoordinateSystem(50.0);
    // Optionally, add labels to the axes
    viewer->addText3D("X", pcl::PointXYZ(1.1, 0, 0), 0.1, 1, 0, 0, "X_axis");
    viewer->addText3D("Y", pcl::PointXYZ(0, 1.1, 0), 0.1, 0, 1, 0, "Y_axis");
    viewer->addText3D("Z", pcl::PointXYZ(0, 0, 1.1), 0.1, 0, 0, 1, "Z_axis");
    // Random number generation for colors
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    // Generate a random color for the cluster
    int r = dis(gen);
    int g = dis(gen);
    int b = dis(gen);

    // Get the bounding box points
    pcl::PointXYZ minPt, maxPt;
    pcl::getMinMax3D(*cloud, minPt, maxPt);
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
    viewer->addLine(p1, p2, r / 255.0, g / 255.0, b / 255.0, "line1_");
    viewer->addLine(p2, p3, r / 255.0, g / 255.0, b / 255.0, "line2_");
    viewer->addLine(p3, p4, r / 255.0, g / 255.0, b / 255.0, "line3_");
    viewer->addLine(p4, p1, r / 255.0, g / 255.0, b / 255.0, "line4_");
    viewer->addLine(p5, p6, r / 255.0, g / 255.0, b / 255.0, "line5_");
    viewer->addLine(p6, p7, r / 255.0, g / 255.0, b / 255.0, "line6_");
    viewer->addLine(p7, p8, r / 255.0, g / 255.0, b / 255.0, "line7_");
    viewer->addLine(p8, p5, r / 255.0, g / 255.0, b / 255.0, "line8_");
    viewer->addLine(p1, p5, r / 255.0, g / 255.0, b / 255.0, "line9_");
    viewer->addLine(p2, p6, r / 255.0, g / 255.0, b / 255.0, "line10_");
    viewer->addLine(p3, p7, r / 255.0, g / 255.0, b / 255.0, "line11_");
    viewer->addLine(p4, p8, r / 255.0, g / 255.0, b / 255.0, "line12_");


    viewer->initCameraParameters();

    // Process events
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100000); // Spin with a more reasonable delay
    }
}

void ply_processor::visualizePointCloudV2(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud, Eigen::Vector4f centroid, Eigen::Matrix3f eigen_vectors) {
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);

    // Visualize original cloud in white
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> white_color(cloud, 255, 255, 255);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, white_color, "original cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "original cloud");

    // Visualize transformed cloud in red
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red_color(transformed_cloud, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(transformed_cloud, red_color, "transformed cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "transformed cloud");

    // Add a coordinate system to the visualizer
    viewer->addCoordinateSystem(50.0);
    // Optionally, add labels to the axes
    viewer->addText3D("X", pcl::PointXYZ(1.1, 0, 0), 0.1, 1, 0, 0, "X_axis");
    viewer->addText3D("Y", pcl::PointXYZ(0, 1.1, 0), 0.1, 0, 1, 0, "Y_axis");
    viewer->addText3D("Z", pcl::PointXYZ(0, 0, 1.1), 0.1, 0, 0, 1, "Z_axis");

    // Draw eigenvectors as arrows
    pcl::PointXYZ origin;
    origin.x = centroid[0];
    origin.y = centroid[1];
    origin.z = centroid[2];
    pcl::PointXYZ end;
    for (int i = 0; i < 3; ++i) {
        end.x = origin.x + 0.1 * eigen_vectors(0, i); // Scale factor for visualization
        end.y = origin.y + 0.1 * eigen_vectors(1, i);
        end.z = origin.z + 0.1 * eigen_vectors(2, i);
        std::string arrow_id = "eigen_vector_" + std::to_string(i);
        viewer->addArrow(end, origin, 1.0, 0.0, 0.0, false, arrow_id);
    }

    // Spin until closed
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100000);
    }
}

