# NMPC Waypoint Planner

<img src="https://github.com/user-attachments/assets/f92ecdd2-ec59-4e00-a447-ab8dfa0f24b8"
     alt="plan"
     width="450">

This repository contains a lightweight C++ waypoint planner designed to generate collision-free intermediate poses (waypoints) for a robot end-effector, in environments modelled using point clouds. The planner relies on an NMPC-style (Nonlinear Model Predictive Control) approach to ensure:
1. Smooth convergence from a start pose to a target pose
2. Avoidance of obstacles
3. Sufficient visibility of the environment (for pose estimation)

---

## Key Features

- **NMPC-Style Optimization**  
  Generates a sequence of intermediate waypoints by minimizing a nonlinear cost function.  
  The cost includes:
  - Pose tracking error  
  - Collision penalty  
  - Visibility constraint  

- **Collision-Aware**  
  Uses [PCL (Point Cloud Library)](http://pointclouds.org/) for nearest-neighbor distance checks and box/frustum culling, ensuring the path remains safe in dense, cluttered scenes.

- **Visibility Constraints**  
  Incorporates field-of-view checks so the camera can maintain a clear view of critical parts of the environment while moving.

- **Dynamic Receding Horizon**  
  Each iteration optimizes over a short horizon, then shifts it forward (like standard NMPC). This makes the planner computationally lighter and more responsive.

---

## Dependencies

- **Eigen** (for linear algebra)
- **PCL** (for point cloud processing, collision checks)
- **nlopt** (for nonlinear optimization, if using the NLopt-based solver)
- **C++17** or above (recommended)


---

## License

This project is licensed under [MIT License](LICENSE). Feel free to modify and adapt the code.

---
