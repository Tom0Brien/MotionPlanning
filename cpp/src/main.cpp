#include <Eigen/Dense>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <pcl/filters/frustum_culling.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "../include/waypoints_planner.hpp"

int main() {
  const size_t StateDim = 6;
  const size_t ActionDim = 6;
  const size_t HorizonDim = 3;
  PlannerMpc<StateDim, ActionDim, HorizonDim, double> planner;

  // 1) Load original cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr originalCloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile("../data/vine_simple.pcd", *originalCloud) == -1) {
    PCL_ERROR("Couldn't read file vine_simple.pcd\n");
    return -1;
  }

  // 2) Downsample
  pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PointXYZ> voxel;
  voxel.setInputCloud(originalCloud);
  voxel.setLeafSize(0.01f, 0.01f, 0.01f);
  voxel.filter(*downsampled_cloud);

  // 3) Build KD-tree
  std::shared_ptr<pcl::KdTreeFLANN<pcl::PointXYZ>> kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZ>);
  kd_tree->setInputCloud(downsampled_cloud);

  // 4) Assign to planner
  planner.obstacle_cloud = downsampled_cloud;
  planner.kd_tree = kd_tree;

  // 5) Set weights, collision, and visibility settings
  /// Positional tracking cost weight.
  planner.w_p = 100.0;
  /// Orientation tracking cost weight.
  planner.w_q = 10.0;
  /// Terminal positional cost weight.
  planner.w_p_term = 1e3;
  /// Terminal orientation cost weight.
  planner.w_q_term = 1e3;
  /// Obstacle avoidance cost weight.
  planner.w_obs = 200.0;
  planner.collision_margin = 0.05;
  planner.w_visibility = 0.0;
  planner.alpha_visibility = 20.0;
  planner.d_thresh = 0.05;
  planner.visibility_fov = 60.0; // degrees
  planner.visibility_min_range = 0.0;
  planner.visibility_max_range = 0.5;

  // 6) Set control bounds
  planner.dp_max = 0.05;
  planner.dp_min = -0.05;
  planner.dtheta_max = 1.0;
  planner.dtheta_min = -1.0;

  // Initial pose
  Eigen::Isometry3d H_0 = Eigen::Isometry3d::Identity();
  H_0.translation() = Eigen::Vector3d(0.0, 0.5, 0.6);
  planner.H_0 = H_0;

  // Goals
  std::vector<Eigen::Isometry3d> goals;
  {
    // Goal 1
    double x1 = 0.23958, y1 = 0.6, z1 = 0.6098;
    double r1 = 0.0, p1 = 0.0, yaw1 = M_PI_2;
    Eigen::AngleAxisd Rz1(yaw1, Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd Ry1(p1, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd Rx1(r1, Eigen::Vector3d::UnitX());
    Eigen::Isometry3d G1 = Eigen::Isometry3d::Identity();
    G1.translation() = Eigen::Vector3d(x1, y1, z1);
    G1.linear() = (Rz1 * Ry1 * Rx1).matrix();
    goals.push_back(G1);

    // Goal 2
    double x2 = 0.53975, y2 = 0.6, z2 = 0.65;
    double r2 = 0.0, p2 = 0.0, yaw2 = M_PI_4;
    Eigen::AngleAxisd Rz2(yaw2, Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd Ry2(p2, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd Rx2(r2, Eigen::Vector3d::UnitX());
    Eigen::Isometry3d G2 = Eigen::Isometry3d::Identity();
    G2.translation() = Eigen::Vector3d(x2, y2, z2);
    G2.linear() = (Rz2 * Ry2 * Rx2).matrix();
    goals.push_back(G2);
  }

  // Write goals to "goals.csv"
  {
    std::string goals_filename = "goals.csv";
    std::ofstream goals_file(goals_filename);
    goals_file << "px,py,pz,roll,pitch,yaw\n";
    for (auto &G : goals) {
      Eigen::Vector3d p = G.translation();
      Eigen::Vector3d eulerZYX = G.rotation().eulerAngles(2, 1, 0);
      double roll = eulerZYX(2);
      double pitch = eulerZYX(1);
      double yaw = eulerZYX(0);
      goals_file << p(0) << "," << p(1) << "," << p(2) << "," << roll << ","
                 << pitch << "," << yaw << "\n";
    }
    goals_file.close();
    std::cout << "[INFO] Wrote " << goals.size() << " goals to "
              << goals_filename << "\n";
  }

  // Container for all waypoints (for each goal)
  std::vector<std::vector<Eigen::Isometry3d>> all_waypoints;

  auto start_total = std::chrono::high_resolution_clock::now();
  std::string filename = "trajectory.csv";
  std::ofstream out(filename);
  out << "px,py,pz,roll,pitch,yaw\n";

  // Generate waypoints for each goal
  for (size_t i = 0; i < goals.size(); ++i) {
    planner.H_0 = (i == 0 ? H_0 : all_waypoints[i - 1].back());
    planner.H_goal = goals[i];

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Eigen::Isometry3d> waypoints =
        planner.generateWaypoints(planner.H_0, planner.H_goal);
    auto end = std::chrono::high_resolution_clock::now();

    all_waypoints.push_back(waypoints);

    std::cout << "\n[Goal " << i << "] Planning took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << " ms. Number of waypoints: " << waypoints.size() << "\n";

    // Write each waypoint to "trajectory.csv"
    for (auto &wp : waypoints) {
      Eigen::Vector3d p = wp.translation();
      Eigen::Vector3d eulerZYX = wp.rotation().eulerAngles(2, 1, 0);
      double roll = eulerZYX(2);
      double pitch = eulerZYX(1);
      double yaw = eulerZYX(0);
      out << p(0) << "," << p(1) << "," << p(2) << "," << roll << "," << pitch
          << "," << yaw << "\n";
    }
  }
  out.close();

  // ------------------------------------------------------------
  // Compute the average number of points visible across the entire plan
  // ------------------------------------------------------------
  {
    double sum_visible_all = 0.0;
    size_t count_waypoints = 0;

    // Loop over all goals' waypoints
    for (const auto &waypoints_for_goal : all_waypoints) {
      for (const auto &wp : waypoints_for_goal) {
        std::size_t visible_count = getVisibleCount(
            planner.obstacle_cloud, planner.visibility_fov,
            planner.visibility_min_range, planner.visibility_max_range, wp);
        sum_visible_all += static_cast<double>(visible_count);
        count_waypoints++;
      }
    }

    if (count_waypoints > 0) {
      double avg_visible_all =
          sum_visible_all / static_cast<double>(count_waypoints);
      std::cout
          << "[Global Plan] Average visible points per waypoint (all goals) = "
          << avg_visible_all << "\n";
    } else {
      std::cout << "[Global Plan] No waypoints generated; cannot compute "
                   "visibility.\n";
    }
  }

  auto end_total = std::chrono::high_resolution_clock::now();
  std::cout << "Total planning for all goals took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   end_total - start_total)
                   .count()
            << " ms.\n";

  return 0;
}
