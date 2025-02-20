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
    const size_t StateDim   = 6;
    const size_t ActionDim  = 6;
    const size_t HorizonDim = 6;
    PlannerMpc<StateDim, ActionDim, HorizonDim, double> planner;

    // 1) Load original cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr originalCloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile("../data/vine_simple_streo_scan.pcd", *originalCloud) == -1) {
        PCL_ERROR("Couldn't read file .pcd\n");
        return -1;
    }

    // 2) Downsample
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setInputCloud(originalCloud);
    voxel.setLeafSize(0.02f, 0.02f, 0.02f);
    voxel.filter(*downsampled_cloud);

    // 3) Build KD-tree
    std::shared_ptr<pcl::KdTreeFLANN<pcl::PointXYZ>> kd_tree(new pcl::KdTreeFLANN<pcl::PointXYZ>);
    kd_tree->setInputCloud(downsampled_cloud);

    // 4) Assign to planner
    planner.obstacle_cloud = downsampled_cloud;
    planner.kd_tree        = kd_tree;

    // 5) Set weights, collision, and visibility settings
    planner.w_p                  = 1e3;  // Positional tracking cost weight.
    planner.w_q                  = 1e2;  // Orientation tracking cost weight.
    planner.w_p_term             = 1e4;  // Terminal positional cost weight.
    planner.w_q_term             = 1e2;  // Terminal orientation cost weight.
    planner.w_obs                = 1e3;  // Obstacle avoidance cost weight.
    planner.collision_margin     = 0.05;
    planner.w_visibility         = 200.0;
    planner.alpha_visibility     = 20.0;
    planner.d_thresh             = 0.05;
    planner.visibility_fov       = 60.0;  // Field of view in degrees.
    planner.visibility_min_range = 0.01;
    planner.visibility_max_range = 0.5;
    planner.max_iterations       = 20;
    planner.min_visible_points   = 20;

    // MPPI parameters
    planner.num_samples   = 200;           // Number of candidate trajectories to sample.
    planner.mppi_lambda   = double(1.0);   // Temperature parameter.
    planner.noise_std_pos = double(0.01);  // Standard deviation for position noise.
    planner.noise_std_ori = double(0.05);  // Standard deviation for orientation noise.

    // Fusion parameters
    planner.fusion_position_tolerance    = 0.02;
    planner.fusion_orientation_tolerance = 0.1;

    // 6) Set control bounds
    planner.dp_max     = 0.025;
    planner.dp_min     = -0.025;
    planner.dtheta_max = 0.1;
    planner.dtheta_min = -0.1;

    // 7) Update the collision box from the STL model of the end effector.
    // Here we load the STL file from "../data/cutter.stl" and apply a rigid transform to it to position it correctly in
    // the camera frame. Modify cutter_transform if a different pose is desired.
    Eigen::Isometry3d cutter_transform = Eigen::Isometry3d::Identity();
    cutter_transform.translation()     = Eigen::Vector3d(0.0, 0.08, 0.03);
    Eigen::AngleAxisd Rzc(M_PI_2, Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd Ryc(0, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd Rxc(M_PI_2, Eigen::Vector3d::UnitX());
    cutter_transform.linear() = (Rzc * Ryc * Rxc).matrix();
    double collision_margin   = 0.005;
    planner.updateEndEffectorFromSTL("../data/cutter.stl", cutter_transform, planner.collision_margin);

    // Save the box dimensions to a file for later visualization.
    std::ofstream box_file("cutter_box.txt");
    if (box_file.is_open()) {
        box_file << planner.box_min.transpose() << "\n" << planner.box_max.transpose() << "\n";
        box_file.close();
        std::cout << "[PlannerMpc::updateEndEffectorBoxFromSTL] Saved box dimensions to cutter_box.txt\n";
    }
    else {
        std::cerr << "[PlannerMpc::updateEndEffectorBoxFromSTL] Failed to open cutter_box.txt for writing\n";
    }

    // Save the relative transform for the cutter (from the camera frame) to a file.
    {
        // 'cutter_transform' is an Eigen::Isometry3d.
        Eigen::Quaterniond q(cutter_transform.rotation());
        Eigen::Vector3d t = cutter_transform.translation();

        std::ofstream transform_file("cutter_transform.txt");
        if (transform_file.is_open()) {
            // Write translation (x y z)
            transform_file << t(0) << " " << t(1) << " " << t(2) << "\n";
            // Write quaternion (qw qx qy qz)
            transform_file << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << "\n";
            transform_file.close();
            std::cout << "[INFO] Saved cutter relative transform to cutter_transform.txt\n";
        }
        else {
            std::cerr << "[ERROR] Could not open cutter_transform.txt for writing.\n";
        }
    }

    // 8) Set the initial pose
    Eigen::Isometry3d H_0 = Eigen::Isometry3d::Identity();
    H_0.translation()     = Eigen::Vector3d(0.2, 0.2, 0.725);
    double r1 = -M_PI_2, p1 = 0, yaw1 = 0;
    Eigen::AngleAxisd Rz0(yaw1, Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd Ry0(p1, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd Rx0(r1, Eigen::Vector3d::UnitX());
    H_0.linear() = (Rz0 * Ry0 * Rx0).matrix();
    // H_0.linear() << 0.6425468638021853, 0.7591589463428162, -0.10393779340133653, 0.07738188639841208,
    //     0.07066251869164698, 0.9944846379633997, 0.7623164161794022, -0.6470583456225703, -0.013341171570557007;
    // H_0.translation() << 0.2930928703755895, 0.4462194264333198, 0.7255900206364525;

    // 9) Define goals
    std::vector<Eigen::Isometry3d> goals;
    {
        // Goal 1
        // double x1 = 0.0, y1 = 0.5, z1 = 0.6;
        // double r1 = -M_PI_2, p1 = 0, yaw1 = 0;
        // Eigen::AngleAxisd Rz1(yaw1, Eigen::Vector3d::UnitZ());
        // Eigen::AngleAxisd Ry1(p1, Eigen::Vector3d::UnitY());
        // Eigen::AngleAxisd Rx1(r1, Eigen::Vector3d::UnitX());
        // Eigen::Isometry3d G1 = Eigen::Isometry3d::Identity();
        // G1.translation()     = Eigen::Vector3d(x1, y1, z1);
        // G1.linear()          = (Rz1 * Ry1 * Rx1).matrix();
        // goals.push_back(G1);
        // goals.push_back(G1);

        // Goal 2
        // double x2 = 0.53975, y2 = 0.6, z2 = 0.725;
        // double r2 = -M_PI_2, p2 = 0, yaw2 = 0;
        // Eigen::AngleAxisd Rz2(yaw2, Eigen::Vector3d::UnitZ());
        // Eigen::AngleAxisd Ry2(p2, Eigen::Vector3d::UnitY());
        // Eigen::AngleAxisd Rx2(r2, Eigen::Vector3d::UnitX());
        // Eigen::Isometry3d G2 = Eigen::Isometry3d::Identity();
        // G2.translation()     = Eigen::Vector3d(x2, y2, z2);
        // G2.linear()          = (Rz2 * Ry2 * Rx2).matrix();
        // goals.push_back(G2);
        // Example: Goal(from the real robot)
        Eigen::Isometry3d H_goal_1;
        H_goal_1.linear() << 0.9907284963734801, 0.01923310699281764, -0.13441684452520866, 0.1337577243239575,
            0.032219489672272936, 0.9904804604618397, 0.02338085880431727, -0.9992957480549294, 0.029348189933893397;
        H_goal_1.translation() << 0.243274508481936, 0.36056617285391136, 0.6743629428023914;

        Eigen::Isometry3d H_goal_2;
        H_goal_2.linear() << 0.6425468638021853, 0.7591589463428162, -0.10393779340133653, 0.07738188639841208,
            0.07066251869164698, 0.9944846379633997, 0.7623164161794022, -0.6470583456225703, -0.013341171570557007;
        H_goal_2.translation() << 0.2930928703755895, 0.4462194264333198, 0.7255900206364525;

        goals.emplace_back(H_goal_1);
        // goals.emplace_back(H_goal_2);
    }

    // Write goals to "goals.csv"
    {
        std::string goals_filename = "goals.csv";
        std::ofstream goals_file(goals_filename);
        goals_file << "px,py,pz,roll,pitch,yaw\n";
        for (const auto& G : goals) {
            Eigen::Vector3d p        = G.translation();
            Eigen::Vector3d eulerZYX = G.rotation().eulerAngles(2, 1, 0);
            double roll              = eulerZYX(2);
            double pitch             = eulerZYX(1);
            double yaw               = eulerZYX(0);
            goals_file << p(0) << "," << p(1) << "," << p(2) << "," << roll << "," << pitch << "," << yaw << "\n";
        }
        goals_file.close();
        std::cout << "[INFO] Wrote " << goals.size() << " goals to " << goals_filename << "\n";
    }

    // Container for all waypoints (for each goal)
    std::vector<std::vector<Eigen::Isometry3d>> all_waypoints;

    auto start_total     = std::chrono::high_resolution_clock::now();
    std::string filename = "trajectory.csv";
    std::ofstream out(filename);
    out << "px,py,pz,roll,pitch,yaw\n";

    // Generate waypoints for each goal
    for (size_t i = 0; i < goals.size(); ++i) {
        // For subsequent goals, use the 2nd last waypoint from the previous segment as the initial condition.
        if (i == 0) {
            planner.H_0 = H_0;
        }
        else {
            const auto& prev_waypoints = all_waypoints[i - 1];
            if (prev_waypoints.size() >= 2) {
                planner.H_0 = prev_waypoints[prev_waypoints.size() - 2];
            }
            else {
                // Fallback to the last waypoint if a 2nd last isn't available.
                planner.H_0 = prev_waypoints.back();
            }
        }
        planner.H_goal = goals[i];

        auto start                               = std::chrono::high_resolution_clock::now();
        std::vector<Eigen::Isometry3d> waypoints = planner.generateWaypoints(planner.H_0, planner.H_goal);

        // waypoints.push_back(waypoints[waypoints.size() - 2]);
        auto end = std::chrono::high_resolution_clock::now();

        all_waypoints.push_back(waypoints);

        std::cout << "\n[Goal " << i << "] Planning took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << " ms. Number of waypoints: " << waypoints.size() << "\n";

        // Write each waypoint to "trajectory.csv"
        for (const auto& wp : waypoints) {
            Eigen::Vector3d p        = wp.translation();
            Eigen::Vector3d eulerZYX = wp.rotation().eulerAngles(2, 1, 0);
            double roll              = eulerZYX(2);
            double pitch             = eulerZYX(1);
            double yaw               = eulerZYX(0);
            out << p(0) << "," << p(1) << "," << p(2) << "," << roll << "," << pitch << "," << yaw << "\n";
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
        for (const auto& waypoints_for_goal : all_waypoints) {
            for (const auto& wp : waypoints_for_goal) {
                std::size_t visible_count = getFrustrumCloud(planner.obstacle_cloud,
                                                             planner.visibility_fov,
                                                             planner.visibility_min_range,
                                                             planner.visibility_max_range,
                                                             wp)
                                                ->size();
                sum_visible_all += static_cast<double>(visible_count);
                count_waypoints++;
            }
        }

        if (count_waypoints > 0) {
            double avg_visible_all = sum_visible_all / static_cast<double>(count_waypoints);
            std::cout << "[Global Plan] Average visible points per waypoint (all goals) = " << avg_visible_all << "\n";
        }
        else {
            std::cout << "[Global Plan] No waypoints generated; cannot compute visibility.\n";
        }
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    std::cout << "Total planning for all goals took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count() << " ms.\n";

    // ------------------------------------------------------------
    // Post-process: extract collision debug points from all waypoints.
    // ------------------------------------------------------------
    pcl::PointCloud<pcl::PointXYZ>::Ptr collision_debug_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    for (const auto& waypoints_for_goal : all_waypoints) {
        for (const auto& wp : waypoints_for_goal) {
            // Extract the points that are inside the collision box at this waypoint.
            pcl::PointCloud<pcl::PointXYZ>::Ptr in_box =
                extractPointsInBox<double>(originalCloud, wp, planner.box_min, planner.box_max);

            // Append these points to the collision debug cloud.
            collision_debug_cloud->points.insert(collision_debug_cloud->points.end(),
                                                 in_box->points.begin(),
                                                 in_box->points.end());
        }
    }

    // Save the debug cloud to a file.
    if (collision_debug_cloud->points.empty()) {
        std::cerr << "[DEBUG] No collision points found in the trajectory.\n";
        // Delete the file if it exists.
        std::filesystem::remove("collision_debug.pcd");
    }
    else {
        collision_debug_cloud->width  = static_cast<uint32_t>(collision_debug_cloud->points.size());
        collision_debug_cloud->height = 1;
        pcl::io::savePCDFileASCII("collision_debug.pcd", *collision_debug_cloud);
        std::cout << "[DEBUG] Saved collision debug cloud with " << collision_debug_cloud->points.size()
                  << " points to collision_debug.pcd" << std::endl;
    }


    return 0;
}
