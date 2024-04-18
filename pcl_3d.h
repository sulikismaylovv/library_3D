#ifndef PCL_3D_H
#define PCL_3D_H

#include "3d_system_global.h"
#include "ply_processor.h"
#include "ply_segmentation.h"

class SYSTEM_3D_EXPORT PCL_3D
{
    std::unique_ptr<ply_segmentation> segmentation;
    std::unique_ptr<ply_processor> processor;

public:
    /**
     * @brief Constructor for the PCL_3D class.
     */
    PCL_3D()
        : segmentation(std::make_unique<ply_segmentation>()),
        processor(std::make_unique<ply_processor>())
    {
        // Empty constructor
    }

    /**
     * @brief Finds the bounding box of a point cloud and returns the eigen vectors representing its orientation.
     *
     * @param filePathBox The file path to the point cloud representing the box.
     * @param filePathTray The file path to the point cloud representing the tray.
     * @param referencePoint The reference point (x, y, z) to be used in calculations.
     * @param prevLocation The previous location (x, y, z) of the point cloud, used for orientation purposes (optional).
     * @return Vector of ClusterInfo objects containing the bounding box information for each cluster.
     */
    std::vector<ClusterInfo> findBoundingBox(const std::string& filePathBox,
                                    const std::string& filePathTray,
                                    const Eigen::Vector3f& referencePoint,
                                    const Eigen::Vector3f& prevLocation,
                                    const Eigen::Vector3f& dimensions = Eigen::Vector3f::Zero());

    /**
     * @brief Calibrates the tray to find the reference point based on the given height and point cloud.
     *
     * @param cloud The point cloud representing the tray.
     * @param height The height at which to find the reference point.
     * @return Eigen::Vector3f The (x, y, z) coordinates of the reference point.
     */
    Eigen::Vector3f calibrateTray(const std::string& filePath,
                                  float height);

    /**
     * @brief Transforms the given point cloud to align with the specified reference point.
     *
     * @param cloud The point cloud to be transformed.
     * @param referencePoint The target reference point (x, y, z) to align the cloud with.
     * @return pcl::PointCloud<pcl::PointXYZ>::Ptr The transformed point cloud.
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformToReferencePoint(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                                                  const Eigen::Vector3f& referencePoint);


    /**
     * @brief preprocessPointCloud function preprocesses the point cloud to find the bounding box and returns the isolated point cloud.
     * @param filePathBox The file path to the point cloud representing the box.
     * @param filePathTray The file path to the point cloud representing the tray.
     * @param referencePoint The reference point (x, y, z) to be used in calculations.
     * @param prevLocation The previous location (x, y, z) of the point cloud, used for orientation purposes (optional).
     * @return Vector of ClusterInfo objects containing the bounding box information for each cluster.
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr preprocessPointCloud(const std::string& filePathBox,
                                                             const std::string &filePathTray,
                                                             const Eigen::Vector3f& referencePoint,
                                                             const Eigen::Vector3f& prevLocation);


    //Light Version of the findBoundingBox function that takes only the isolated PCL and ref point
    std::vector<ClusterInfo> findBoundingBoxLight(const pcl::PointCloud<pcl::PointXYZ>::Ptr& isolated_pcl,
                                                  const Eigen::Vector3f& referencePoint);
};

#endif // PCL_3D_H
