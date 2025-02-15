#ifndef WAYPOINT_PLANNER_HPP
#define WAYPOINT_PLANNER_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <nlopt.hpp>
#include <pcl/filters/frustum_culling.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>

/**
 * @brief Builds a 4x4 homogeneous transform from a position and ZYX Euler
 * angles.
 *
 * @tparam Scalar The scalar type.
 * @param p The position vector.
 * @param eul The Euler angles (roll, pitch, yaw) in ZYX order.
 * @return An IsometryT representing the homogeneous transform.
 */
template <typename Scalar>
Eigen::Transform<Scalar, 3, Eigen::Isometry> eulerZYXToIsometry(const Eigen::Matrix<Scalar, 3, 1>& translation,
                                                                const Eigen::Matrix<Scalar, 3, 1>& eulZYX) {
    Eigen::Transform<Scalar, 3, Eigen::Isometry> T = Eigen::Transform<Scalar, 3, Eigen::Isometry>::Identity();
    T.linear() = Eigen::AngleAxis<Scalar>(eulZYX.z(), Eigen::Matrix<Scalar, 3, 1>::UnitZ())
                 * Eigen::AngleAxis<Scalar>(eulZYX.y(), Eigen::Matrix<Scalar, 3, 1>::UnitY())
                 * Eigen::AngleAxis<Scalar>(eulZYX.x(), Eigen::Matrix<Scalar, 3, 1>::UnitX()).toRotationMatrix();
    T.translation() = translation;
    return T;
}

/**
 * @brief Computes the 6D homogeneous error between two transforms.
 *
 * The error vector consists of 6 elements where the first 3 represent the
 * translational error and the last 3 represent the rotational error.
 *
 * @tparam Scalar The scalar type.
 * @param H1 The first homogeneous transform.
 * @param H2 The second homogeneous transform.
 * @return A 6x1 Eigen vector representing the error.
 */
template <typename Scalar>
Eigen::Matrix<Scalar, 6, 1> homogeneousError(const Eigen::Transform<Scalar, 3, Eigen::Isometry>& H1,
                                             const Eigen::Transform<Scalar, 3, Eigen::Isometry>& H2) {
    Eigen::Matrix<Scalar, 6, 1> e = Eigen::Matrix<Scalar, 6, 1>::Zero();

    // Translational error
    e.head(3) = H1.translation() - H2.translation();

    // Orientation error
    Eigen::Matrix<Scalar, 3, 3> Re = H1.rotation() * H2.rotation().transpose();
    Scalar t                       = Re.trace();
    Eigen::Matrix<Scalar, 3, 1> eps(Re(2, 1) - Re(1, 2), Re(0, 2) - Re(2, 0), Re(1, 0) - Re(0, 1));
    Scalar eps_norm = eps.norm();
    if (t > -0.99 || eps_norm > 1e-10) {
        if (eps_norm < 1e-3)
            e.tail(3) = (Scalar(0.75) - t / Scalar(12)) * eps;
        else
            e.tail(3) = (std::atan2(eps_norm, t - Scalar(1)) / eps_norm) * eps;
    }
    else {
        e.tail(3) = M_PI_2 * (Re.diagonal().array() + Scalar(1));
    }
    return e;
}

/**
 * @brief Helper function to compute the number of points visible from a given
 * pose under the new coordinate mapping (X forward, Y left, Z up).
 *
 * @param cloud      The (downsampled) obstacle cloud.
 * @param fov_degs   Field of view in degrees (horizontal & vertical).
 * @param near_plane Near plane distance for frustum culling.
 * @param far_plane  Far plane distance for frustum culling.
 * @param pose       The pose in world coordinates (where we consider X forward,
 * Y left, Z up).
 * @return The number of points in the cloud that lie within the camera frustum.
 */
template <typename Scalar>
std::size_t getVisibleCount(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud,
                            Scalar fov_degs,
                            Scalar near_plane,
                            Scalar far_plane,
                            const Eigen::Transform<Scalar, 3, Eigen::Isometry>& pose) {
    if (!cloud || cloud->points.empty()) {
        return 0;
    }

    // Create a FrustumCulling filter
    pcl::FrustumCulling<pcl::PointXYZ> fc;
    fc.setInputCloud(cloud);
    fc.setHorizontalFOV(static_cast<float>(fov_degs));
    fc.setVerticalFOV(static_cast<float>(fov_degs));
    fc.setNearPlaneDistance(static_cast<float>(near_plane));
    fc.setFarPlaneDistance(static_cast<float>(far_plane));

    // Rotate camera so that PCL "camera" aligns with X forward, Y left, Z up.
    Eigen::Matrix4f camera_pose = pose.matrix().template cast<float>();
    Eigen::Matrix4f cam2robot;
    cam2robot << 1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1;
    // cam2robot << 0, 0, 1, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1; // X right, Y
    // down, Z forward
    camera_pose = camera_pose * cam2robot;

    // Apply the transform in PCL
    fc.setCameraPose(camera_pose);

    // Filter the points that lie within this frustum
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>());
    fc.filter(*cloud_out);

    return cloud_out->size();
}

/**
 * @brief PlannerMpc class implementing an NLMPC-style trajectory planner.
 *
 * This class is templated on the state dimension (StateDim), action dimension
 * (ActionDim), horizon (HorizonDim), and the scalar type (Scalar). For the
 * simple integrator used here, it is assumed that StateDim == ActionDim, and
 * that the first 3 entries of the state vector represent the position while the
 * last 3 represent the Euler angles.
 *
 * @tparam StateDim The dimension of the state vector.
 * @tparam ActionDim The dimension of the action (control) vector.
 * @tparam HorizonDim The planning horizon.
 * @tparam Scalar The scalar type (default is double).
 */
template <int StateDim, int ActionDim, int HorizonDim, typename Scalar = double>
class PlannerMpc {
public:
    /// Type alias for the isometry using the specified scalar type.
    using IsometryT = Eigen::Transform<Scalar, 3, Eigen::Isometry>;

    /// Initial pose.
    IsometryT H_0 = IsometryT::Identity();
    /// Goal pose.
    IsometryT H_goal = IsometryT::Identity();

    /// Positional tracking cost weight.
    Scalar w_p = Scalar(100.0);
    /// Orientation tracking cost weight.
    Scalar w_q = Scalar(10.0);
    /// Terminal positional cost weight.
    Scalar w_p_term = Scalar(1e3);
    /// Terminal orientation cost weight.
    Scalar w_q_term = Scalar(1e3);

    /// Visibility cost weight.
    Scalar w_visibility = Scalar(20.0);
    /// Tuning parameter to control the saturation rate.
    Scalar alpha_visibility = Scalar(20.0);
    /// Field of view (in degrees) for visibility checking.
    Scalar visibility_fov = Scalar(60.0);
    /// Min plane distances for the visibility frustum.
    Scalar visibility_min_range = Scalar(0.0);
    /// Max plane distance for the visibility frustum.
    Scalar visibility_max_range = Scalar(0.5);
    /// Distance from goal to begin diminishing visibility cost term
    Scalar d_thresh = 0.1;

    /// Obstacle avoidance cost weight.
    Scalar w_obs = Scalar(5.0);
    /// Obstacle cloud for avoidance and visibility.
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr obstacle_cloud;
    /// KD-tree for obstacle queries.
    std::shared_ptr<pcl::KdTreeFLANN<pcl::PointXYZ>> kd_tree;
    /// Safety margin for collision avoidance.
    Scalar collision_margin = Scalar(0.05);

    /// Control bounds for position.
    Scalar dp_min = Scalar(-0.1);
    Scalar dp_max = Scalar(0.1);
    /// Control bounds for orientation.
    Scalar dtheta_min = Scalar(-0.1);
    Scalar dtheta_max = Scalar(0.1);

    /// Maintained control sequence for warm-starting (size: ActionDim *
    /// HorizonDim).
    std::vector<Scalar> U;

    /// Convergence criteria for waypoint generation: position tolerance (1 mm).
    double position_tolerance = 1e-2;
    /// Convergence criteria for waypoint generation: orientation tolerance
    /// (~0.057 deg).
    double orientation_tolerance = 1e-2;
    /// Maximum number of iterations for waypoint generation.
    int max_iterations = 100;

    /**
     * @brief Default constructor.
     */
    PlannerMpc() = default;

    /**
     * @brief Default destructor.
     */
    ~PlannerMpc() = default;

    /**
     * @brief Sets a new warm-start control sequence.
     *
     * @param U_init The initial control sequence.
     */
    void setAction(const std::vector<Scalar>& U_init) {
        if (static_cast<int>(U_init.size()) == ActionDim * HorizonDim) {
            U = U_init;
        }
        else {
            std::cerr << "[PlannerMpc::setAction] Mismatched size! Expected " << (ActionDim * HorizonDim) << ".\n";
        }
    }

    /**
     * @brief Rollouts the trajectory using the given control sequence.
     *
     * Converts a std::vector<Scalar> control sequence (length = ActionDim *
     * HorizonDim) into a dynamic Eigen representation and integrates the
     * trajectory starting from H_0. Each state is represented as an
     * Eigen::Matrix<Scalar, StateDim, 1>.
     *
     * @param U_in The control sequence.
     * @return A vector of state vectors representing the trajectory.
     */
    std::vector<Eigen::Matrix<Scalar, StateDim, 1>> rollout(const std::vector<Scalar>& U_in) {
        std::vector<Eigen::Matrix<Scalar, StateDim, 1>> trajectory(HorizonDim + 1);
        // Convert H_0 into a state vector: first 3 entries: translation; next 3:
        // Euler angles (from ZYX).
        Eigen::Matrix<Scalar, StateDim, 1> s0;
        Eigen::Matrix<Scalar, 3, 1> p0       = H_0.translation();
        Eigen::Matrix<Scalar, 3, 1> eulerZYX = H_0.rotation().eulerAngles(2, 1, 0);
        // Reorder to (x,y,z): (eulerZYX(2), eulerZYX(1), eulerZYX(0))
        Eigen::Matrix<Scalar, 3, 1> eul0;
        eul0 << eulerZYX(2), eulerZYX(1), eulerZYX(0);
        s0 << p0, eul0;
        trajectory[0] = s0;

        // Convert U_in into an Eigen matrix (ActionDim x HorizonDim)
        Eigen::Matrix<Scalar, ActionDim, HorizonDim> U_mat;
        for (int k = 0; k < HorizonDim; ++k) {
            for (int i = 0; i < ActionDim; ++i) {
                U_mat(i, k) = U_in[ActionDim * k + i];
            }
        }
        // For a simple integrator: next state = current state + control.
        for (int k = 0; k < HorizonDim; ++k) {
            trajectory[k + 1] = trajectory[k] + U_mat.col(k);
        }
        return trajectory;
    }

    /**
     * @brief Computes the obstacle cost based on the distance from a query point.
     *
     * @param p The position at which to evaluate the obstacle cost.
     * @return The computed obstacle cost.
     */
    Scalar obstacleCost(const Eigen::Matrix<Scalar, 3, 1>& p) {
        if (kd_tree && kd_tree->getInputCloud() && !kd_tree->getInputCloud()->points.empty()) {
            pcl::PointXYZ query_pt;
            query_pt.x = static_cast<float>(p(0));
            query_pt.y = static_cast<float>(p(1));
            query_pt.z = static_cast<float>(p(2));
            std::vector<int> nn_index(1);
            std::vector<float> nn_dist2(1);
            int found = kd_tree->nearestKSearch(query_pt, 1, nn_index, nn_dist2);
            if (found > 0) {
                Scalar nearest_dist = std::sqrt(nn_dist2[0]);
                if (nearest_dist < collision_margin) {
                    Scalar diff = (Scalar(1.0) / nearest_dist) - (Scalar(1.0) / collision_margin);
                    return Scalar(0.5) * w_obs * diff * diff;
                }
            }
        }
        return Scalar(0.0);
    }

    /**
     * @brief Computes the pose cost based on the error relative to the goal pose.
     *
     * @param p The current position vector.
     * @param eul The current Euler angles.
     * @param wp The weight for positional error.
     * @param wq The weight for orientation error.
     * @return The computed pose cost.
     */
    Scalar poseCost(const Eigen::Matrix<Scalar, 3, 1>& p,
                    const Eigen::Matrix<Scalar, 3, 1>& eul,
                    Scalar wp,
                    Scalar wq) {
        auto H = eulerZYXToIsometry<Scalar>(p, eul);
        auto e = homogeneousError(H, H_goal);
        return wp * e.head(3).squaredNorm() + wq * e.tail(3).squaredNorm();
    }

    /**
     * @brief Computes a visibility cost based on the fraction of obstacle points
     *        that lie within the camera frustum of the current pose (X forward, Y
     * left, Z up).
     *
     * @param p   The current position (x,y,z).
     * @param eul The current ZYX Euler angles (roll, pitch, yaw).
     * @return The computed visibility cost (lower cost for higher visibility).
     */
    Scalar visibilityCost(const Eigen::Matrix<Scalar, 3, 1>& p, const Eigen::Matrix<Scalar, 3, 1>& eul) {
        if (!obstacle_cloud || obstacle_cloud->points.empty())
            return Scalar(0);

        auto H = eulerZYXToIsometry<Scalar>(p, eul);
        std::size_t visible =
            getVisibleCount(obstacle_cloud, visibility_fov, visibility_min_range, visibility_max_range, H);
        Scalar ratio = static_cast<Scalar>(visible) / static_cast<Scalar>(obstacle_cloud->points.size());

        // Reduce importance of this term if near the goal
        Scalar dist  = (p - H_goal.translation()).norm();
        Scalar scale = std::min(Scalar(1), dist / d_thresh);

        // Exponential reward => cost is high if ratio is low
        return w_visibility * std::exp(-alpha_visibility * ratio) * scale;
    }

    /**
     * @brief Computes the total cost along the trajectory induced by the control
     * sequence.
     *
     * @param x The control sequence.
     * @param grad The gradient of the cost (if required).
     * @return The total cost.
     */
    Scalar cost(const std::vector<Scalar>& x, std::vector<Scalar>& grad) {
        auto traj   = rollout(x);
        Scalar cost = 0;
        for (int k = 0; k <= HorizonDim; ++k) {
            auto p          = traj[k].template head<3>();
            auto eul        = traj[k].template tail<3>();
            double box_cost = boxCollisionCost(eulerZYXToIsometry<Scalar>(p, eul));
            std::cout << "[PlannerMpc::cost] Box collision cost: " << box_cost << std::endl;
            cost += poseCost(p, eul, w_p, w_q) + obstacleCost(p) + visibilityCost(p, eul);
        }
        // Terminal cost
        auto p_N   = traj[HorizonDim].template head<3>();
        auto eul_N = traj[HorizonDim].template tail<3>();
        cost += poseCost(p_N, eul_N, w_p_term, w_q_term);
        return cost;
    }

    /**
     * @brief Static cost wrapper for NLopt callback.
     *
     * @param x The control sequence.
     * @param grad The gradient of the cost.
     * @param data Pointer to the PlannerMpc instance.
     * @return The cost computed by the PlannerMpc instance.
     */
    static Scalar costWrapper(const std::vector<Scalar>& x, std::vector<Scalar>& grad, void* data) {
        PlannerMpc* planner_ptr = reinterpret_cast<PlannerMpc*>(data);
        return planner_ptr->cost(x, grad);
    }

    /**
     * @brief Solves the MPC problem using NLopt and returns the optimized control
     * sequence.
     *
     * @param H0_in The initial pose.
     * @return The optimized control sequence.
     */
    std::vector<Scalar> getAction(const IsometryT& H0_in) {
        H_0 = H0_in;
        if (HorizonDim <= 0) {
            std::cerr << "[PlannerMpc::getAction] HorizonDim <= 0.\n";
            return {};
        }
        if (static_cast<int>(U.size()) != ActionDim * HorizonDim)
            U.assign(ActionDim * HorizonDim, Scalar(0));

        int dim = ActionDim * HorizonDim;
        nlopt::opt localOpt(nlopt::LN_BOBYQA, dim);
        nlopt::opt opt(nlopt::AUGLAG, dim);
        opt.set_local_optimizer(localOpt);
        opt.set_min_objective(costWrapper, this);

        // Bounds
        std::vector<Scalar> lb(dim), ub(dim);
        for (int k = 0; k < HorizonDim; ++k) {
            // position deltas
            for (int i = 0; i < 3; ++i) {
                lb[ActionDim * k + i] = dp_min;
                ub[ActionDim * k + i] = dp_max;
            }
            // orientation deltas
            for (int i = 3; i < 6; ++i) {
                lb[ActionDim * k + i] = dtheta_min;
                ub[ActionDim * k + i] = dtheta_max;
            }
        }
        opt.set_lower_bounds(lb);
        opt.set_upper_bounds(ub);
        opt.set_xtol_rel(1e-6);
        opt.set_maxeval(100);

        std::vector<Scalar> U_opt = U;  // warm start
        Scalar minf               = 0;
        try {
            auto result = opt.optimize(U_opt, minf);
            std::cout << "[PlannerMpc::getAction] Converged. Cost = " << minf << " (nlopt code: " << result << ")\n";
        }
        catch (std::exception& e) {
            std::cerr << "[PlannerMpc::getAction] NLopt failed: " << e.what() << std::endl;
        }

        // Check final pose error
        auto traj  = rollout(U_opt);
        auto p_N   = traj[HorizonDim].template head<3>();
        auto eul_N = traj[HorizonDim].template tail<3>();
        auto H_N   = eulerZYXToIsometry<Scalar>(p_N, eul_N);
        auto err   = homogeneousError(H_N, H_goal);
        std::cout << "[PlannerMpc::getAction] Final pos error: " << err.head(3).norm()
                  << ", ori error: " << err.tail(3).norm() << "\n";

        // Recede horizon
        if (HorizonDim > 1) {
            for (int k = 0; k < HorizonDim - 1; ++k)
                for (int i = 0; i < ActionDim; ++i)
                    U[ActionDim * k + i] = U_opt[ActionDim * (k + 1) + i];
            std::fill(U.end() - ActionDim, U.end(), Scalar(0));
        }
        else {
            std::fill(U.begin(), U.end(), Scalar(0));
        }

        return U_opt;
    }
    /**
     * @brief Generates waypoints by running the MPC loop from the initial pose to
     * the goal pose, while also computing time statistics and visibility metrics.
     *
     * Runs the MPC loop starting from the initial pose until the convergence
     * criteria or the maximum number of iterations is reached. Returns a vector
     * of waypoints representing the trajectory.
     *
     * @param init The initial pose.
     * @param goal The goal pose.
     * @return A vector of IsometryT waypoints representing the planned
     *         trajectory.
     */
    std::vector<IsometryT> generateWaypoints(const IsometryT& init, const IsometryT& goal) {
        // Start timer
        auto start_time = std::chrono::high_resolution_clock::now();

        H_0    = init;
        H_goal = goal;

        std::vector<IsometryT> waypoints{H_0};
        for (int iter = 0; iter < max_iterations; ++iter) {
            auto U_opt                      = getAction(H_0);
            auto states                     = rollout(U_opt);
            auto next_s                     = states[1];  // receding-horizon step
            Eigen::Matrix<Scalar, 3, 1> p   = next_s.head(3);
            Eigen::Matrix<Scalar, 3, 1> eul = next_s.tail(3);

            IsometryT H_next = eulerZYXToIsometry(p, eul);
            auto err         = homogeneousError(H_next, H_goal);
            double pos_err   = err.head(3).norm();
            double ori_err   = err.tail(3).norm();

            std::cout << "[PlannerMpc::generateWaypoints] Iter " << (iter + 1) << " -> pos_err=" << pos_err
                      << ", ori_err=" << ori_err << "\n";

            if (pos_err < position_tolerance && ori_err < orientation_tolerance) {
                waypoints.back() = H_goal;  // Snap final
                break;
            }
            else {
                H_0 = H_next;
                waypoints.push_back(H_0);
            }
        }

        // End timer
        auto end_time = std::chrono::high_resolution_clock::now();
        auto planning_duration_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        // Compute visibility metric across all generated waypoints
        double sum_visible = 0.0;
        if (obstacle_cloud && !obstacle_cloud->points.empty()) {
            for (const auto& wp : waypoints) {
                std::size_t visible_count =
                    getVisibleCount(obstacle_cloud, visibility_fov, visibility_min_range, visibility_max_range, wp);
                sum_visible += static_cast<double>(visible_count);
            }
        }
        double avg_visible_per_waypoint =
            (waypoints.empty()) ? 0.0 : (sum_visible / static_cast<double>(waypoints.size()));

        // Print out the results
        std::cout << "[PlannerMpc::generateWaypoints] "
                  << "Planning took " << planning_duration_ms << " ms. "
                  << "Number of waypoints: " << waypoints.size() << std::endl;
        std::cout << "[PlannerMpc::generateWaypoints] "
                  << "Average visible points per waypoint: " << avg_visible_per_waypoint << std::endl;

        return waypoints;
    }
};

#endif  // WAYPOINT_PLANNER_HPP
