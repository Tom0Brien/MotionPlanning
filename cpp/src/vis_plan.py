# visualize_trajectory.py

import open3d as o3d
import numpy as np

def load_trajectory_euler(filename):
    """
    Loads the trajectory from a CSV file with lines of:
       px, py, pz, roll, pitch, yaw
    Returns:
       positions: Nx3 numpy array of (px, py, pz)
       eulers:    Nx3 numpy array of (roll, pitch, yaw) in ZYX order
                  i.e. we interpret them as Rz(yaw)*Ry(pitch)*Rx(roll).
    """
    data = np.loadtxt(filename, delimiter=",", skiprows=1)  # skip header
    positions = data[:, 0:3]
    eulers = data[:, 3:6]
    return positions, eulers

def load_goals_euler(filename):
    """
    Similar to load_trajectory_euler but for goals.csv.
    Returns Nx3 positions and Nx3 eulers.
    """
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    positions = data[:, 0:3]
    eulers = data[:, 3:6]
    return positions, eulers

def load_obstacles(filename):
    """
    Loads spherical obstacles from a CSV file with lines of:
       cx, cy, cz, radius
    Returns a list of (center, radius) pairs.
    """
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    obstacles = []
    for row in data:
        cx, cy, cz, r = row
        center = np.array([cx, cy, cz])
        obstacles.append((center, r))
    return obstacles

def eulerZYXToQuaternion(rx, ry, rz):
    """
    Converts ZYX Euler angles (roll=rx, pitch=ry, yaw=rz) into quaternion (qw, qx, qy, qz).
    R = Rz(rz) * Ry(ry) * Rx(rx).
    """
    sx, cx = np.sin(rx), np.cos(rx)
    sy, cy = np.sin(ry), np.cos(ry)
    sz, cz = np.sin(rz), np.cos(rz)

    # Rotation matrices about X, Y, Z
    Rx = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx,  cx]
    ])
    Ry = np.array([
        [ cy, 0, sy],
        [  0, 1,  0],
        [-sy, 0, cy]
    ])
    Rz = np.array([
        [cz, -sz, 0],
        [sz,  cz, 0],
        [ 0,   0, 1]
    ])

    R = Rz @ Ry @ Rx

    # Convert to quaternion
    trace = np.trace(R)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2,1] - R[1,2]) * s
        qy = (R[0,2] - R[2,0]) * s
        qz = (R[1,0] - R[0,1]) * s
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            qw = (R[2,1] - R[1,2]) / s
            qx = 0.25 * s
            qy = (R[0,1] + R[1,0]) / s
            qz = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            qw = (R[0,2] - R[2,0]) / s
            qx = (R[0,1] + R[1,0]) / s
            qy = 0.25 * s
            qz = (R[1,2] + R[2,1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            qw = (R[1,0] - R[0,1]) / s
            qx = (R[0,2] + R[2,0]) / s
            qy = (R[1,2] + R[2,1]) / s
            qz = 0.25 * s

    norm_q = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw /= norm_q
    qx /= norm_q
    qy /= norm_q
    qz /= norm_q

    return np.array([qw, qx, qy, qz])

def create_line_set_from_points(points):
    """
    Creates an Open3D LineSet object connecting consecutive points: (p0->p1->p2->...).
    """
    if len(points) < 2:
        return o3d.geometry.LineSet()
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    lines = [[i, i+1] for i in range(len(points)-1)]
    line_set.lines = o3d.utility.Vector2iVector(lines)
    colors = [[1, 0, 0] for _ in lines]  # red
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def create_coordinate_frame(origin, quaternion, scale=0.1):
    """
    Creates a small coordinate frame at the given origin, oriented by quaternion (qw, qx, qy, qz).
    """
    q_w, q_x, q_y, q_z = quaternion
    # Normalized rotation matrix from quaternion
    norm_q = np.sqrt(q_w**2 + q_x**2 + q_y**2 + q_z**2)
    q_w, q_x, q_y, q_z = q_w / norm_q, q_x / norm_q, q_y / norm_q, q_z / norm_q

    R = np.array([
        [1 - 2*(q_y**2 + q_z**2),     2*(q_x*q_y - q_z*q_w),       2*(q_x*q_z + q_y*q_w)],
        [2*(q_x*q_y + q_z*q_w),       1 - 2*(q_x**2 + q_z**2),     2*(q_y*q_z - q_x*q_w)],
        [2*(q_x*q_z - q_y*q_w),       2*(q_y*q_z + q_x*q_w),       1 - 2*(q_x**2 + q_y**2)]
    ])

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale, origin=[0,0,0])
    frame.rotate(R, center=(0, 0, 0))
    frame.translate(origin)
    return frame

def create_sphere_at_position(position, radius=0.02, color=(0.0, 1.0, 0.0)):
    """
    Creates a sphere (TriangleMesh) at the given position with the specified color.
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
    sphere.paint_uniform_color(color)
    sphere.translate(position)
    return sphere

def load_point_cloud(filename):
    """
    Loads a 3D point cloud from a .pcd file.
    """
    pcd = o3d.io.read_point_cloud(filename)
    if not pcd.has_points():
        raise ValueError(f"Point cloud file '{filename}' is empty or invalid.")
    return pcd

if __name__ == "__main__":
    # ----------------------
    # Load the trajectory
    # ----------------------
    positions, eulers = load_trajectory_euler("trajectory.csv")
    print("Loaded trajectory with", positions.shape[0], "points.")

    # Convert Euler angles to quaternions
    quaternions = np.array([eulerZYXToQuaternion(r, p, y) for (r, p, y) in eulers])

    # Create a line set for the trajectory
    line_set = create_line_set_from_points(positions)

    # Create small frames at each waypoint
    frames = [create_coordinate_frame(pos, quat, scale=0.03)
              for pos, quat in zip(positions, quaternions)]

    # Start and goal spheres (for the entire trajectory):
    # - The "start" is the very first position
    # - The "final" is the very last position
    start_sphere = create_sphere_at_position(positions[0], radius=0.015, color=(0, 1, 0))  # green
    final_sphere = create_sphere_at_position(positions[-1], radius=0.015, color=(1, 0, 0)) # red

    # ----------------------
    # Load the goals
    # ----------------------
    goal_positions = []
    goal_frames = []
    try:
        goals_pos, goals_eulers = load_goals_euler("goals.csv")
        print("Loaded", goals_pos.shape[0], "goals.")
        for i in range(goals_pos.shape[0]):
            gpos = goals_pos[i]
            geul = goals_eulers[i]  # roll, pitch, yaw
            gquat = eulerZYXToQuaternion(geul[0], geul[1], geul[2])
            # Create small sphere or frame for each goal
            # Optional: color-code them differently
            sphere = create_sphere_at_position(gpos, radius=0.01, color=(0.0, 0.5, 1.0))
            frame  = create_coordinate_frame(gpos, gquat, scale=0.04)
            goal_positions.append(sphere)
            goal_frames.append(frame)
    except OSError:
        print("No 'goals.csv' file found or unable to read. Skipping goal visualization.")

    # ----------------------
    # Load obstacles (optional)
    # ----------------------
    obstacle_spheres = []
    try:
        obstacles = load_obstacles("obstacles.csv")
        print(f"Loaded {len(obstacles)} obstacles.")
        for (center, radius) in obstacles:
            sphere = create_sphere_at_position(center, radius=radius, color=(0.0, 0.0, 1.0))
            obstacle_spheres.append(sphere)
    except OSError:
        print("No 'obstacles.csv' found or unable to read. Skipping obstacle visualization.")

    # ----------------------
    # Load the 3D point cloud
    # ----------------------
    point_cloud = None
    try:
        point_cloud = load_point_cloud("../data/vine_simple.pcd")
        print(f"Loaded point cloud with {len(point_cloud.points)} points.")
    except Exception as e:
        print(f"Failed to load point cloud: {e}")

    # Combine all geometries
    geometries = [line_set, start_sphere, final_sphere] + frames
    geometries += goal_positions + goal_frames
    geometries += obstacle_spheres
    if point_cloud:
        geometries.append(point_cloud)

    # Visualize in Open3D
    o3d.visualization.draw_geometries(geometries)
