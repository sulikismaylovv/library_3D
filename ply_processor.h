#ifndef PLY_PROCESSOR_H
#define PLY_PROCESSOR_H

// PCL common functionalities
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

// PCL point cloud handling
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// PCL filters
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

// PCL IO
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

// PCL segmentation
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>

// PCL surface
#include <pcl/surface/mls.h>

// PCL visualization (if needed)
#include <pcl/visualization/pcl_visualizer.h>

// Eigen library
#include <Eigen/Dense>

//KDTree
#include <pcl/search/kdtree.h>

#include <optional>



/**
 * @brief The ply_processor class provides functionalities to process point clouds.
 */
class ply_processor {
public:
    ply_processor() = default;

    /**
     * Loads a point cloud from a file. The file format is auto-detected based on extension.
     * Supported formats: PLY, PCD.
     * @param filePath Path to the file.
     * @return A pointer to the loaded point cloud, or nullptr if loading failed.
     */
    std::optional<pcl::PointCloud<pcl::PointXYZ>::Ptr> loadCloud(const std::string& filePath);

    /**
     * Inverts the Y and Z coordinates of all points in the point cloud.
     * @param cloud Cloud to modify.
     * @return True if the operation was successful, false otherwise.
     */
    bool invertPointCloud(pcl::PointCloud<pcl::PointXYZ>& cloud);

    /**
     * Applies a voxel grid filter to downsample the point cloud.
     * @param cloud Cloud to modify.
     * @param leafSize Size of the voxel grid.
     * @return True if the operation was successful, false otherwise.
     */
    bool applyVoxelGridFilter(pcl::PointCloud<pcl::PointXYZ>& cloud, float leafSize);

    /**
     * Applies a pass-through filter to remove points outside a specified range.
     * @param cloud Cloud to modify.
     * @param minPt Minimum point of the filtering box.
     * @param maxPt Maximum point of the filtering box.
     * @return True if the operation was successful, false otherwise.
     */
    bool applyPassthroughFilter(pcl::PointCloud<pcl::PointXYZ>& cloud, const pcl::PointXYZ& minPt, const pcl::PointXYZ& maxPt);

    /**
     * Applies MLS surface reconstruction to smooth and resample the point cloud.
     * @param cloud Cloud to modify.
     * @param searchRadius Radius used for the MLS search.
     * @return True if the operation was successful, false otherwise.
     */
    bool applyMLSSurfaceReconstruction(pcl::PointCloud<pcl::PointXYZ>& cloud, float searchRadius);

    /**
     * Removes statistical outliers from the point cloud.
     * @param cloud Cloud to modify.
     * @param meanK Number of neighbors to consider for each point.
     * @param stddevMulThresh Standard deviation multiplier for determining which points are considered outliers.
     * @return True if the operation was successful, false otherwise.
     */
    bool applyStatisticalOutlierRemoval(pcl::PointCloud<pcl::PointXYZ>& cloud, int meanK, double stddevMulThresh);

private:
         // Internal helper methods and variables can be declared here
};


#endif // PLY_PROCESSOR_H
