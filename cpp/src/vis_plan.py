import open3d as o3d
import numpy as np
import time
import copy

def load_trajectory_euler(filename):
    """
    Loads the trajectory from a CSV file with lines of:
       px, py, pz, roll, pitch, yaw
    Returns:
       positions: Nx3 numpy array of (px, py, pz)
       eulers:    Nx3 numpy array of (roll, pitch, yaw) in ZYX order,
                  i.e. we interpret them as Rz(yaw)*Ry(pitch)*Rx(roll).
    """
    data = np.loadtxt(filename, delimiter=",", skiprows=1)  # skip header
    positions = data[:, 0:3]
    eulers = data[:, 3:6]
    return positions, eulers

def load_goals_euler(filename):
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    # Ensure data is 2D (in case there's only one goal)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    positions = data[:, 0:3]
    eulers = data[:, 3:6]
    return positions, eulers


def load_obstacles(filename):
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    obstacles = []
    for row in data:
        cx, cy, cz, r = row
        center = np.array([cx, cy, cz])
        obstacles.append((center, r))
    return obstacles

def eulerZYXToQuaternion(rx, ry, rz):
    sx, cx = np.sin(rx), np.cos(rx)
    sy, cy = np.sin(ry), np.cos(ry)
    sz, cz = np.sin(rz), np.cos(rz)

    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx,  cx]])
    Ry = np.array([[ cy, 0, sy],
                   [  0, 1,  0],
                   [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [ 0,   0, 1]])
    R = Rz @ Ry @ Rx

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
    return np.array([qw, qx, qy, qz]) / norm_q

def create_line_set_from_points(points):
    if len(points) < 2:
        return o3d.geometry.LineSet()
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    lines = [[i, i+1] for i in range(len(points)-1)]
    line_set.lines = o3d.utility.Vector2iVector(lines)
    colors = [[1, 0, 0] for _ in lines]
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def create_coordinate_frame(origin, quaternion, scale=0.1):
    q_w, q_x, q_y, q_z = quaternion
    norm_q = np.sqrt(q_w**2 + q_x**2 + q_y**2 + q_z**2)
    q_w, q_x, q_y, q_z = q_w/norm_q, q_x/norm_q, q_y/norm_q, q_z/norm_q
    R = np.array([
        [1 - 2*(q_y**2+q_z**2), 2*(q_x*q_y - q_z*q_w), 2*(q_x*q_z+q_y*q_w)],
        [2*(q_x*q_y+q_z*q_w), 1 - 2*(q_x**2+q_z**2), 2*(q_y*q_z-q_x*q_w)],
        [2*(q_x*q_z-q_y*q_w), 2*(q_y*q_z+q_x*q_w), 1-2*(q_x**2+q_y**2)]
    ])
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale, origin=[0,0,0])
    frame.rotate(R, center=(0,0,0))
    frame.translate(origin)
    return frame

def create_sphere_at_position(position, radius=0.02, color=(0.0, 1.0, 0.0)):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
    sphere.paint_uniform_color(color)
    sphere.translate(position)
    return sphere

def load_point_cloud(filename):
    pcd = o3d.io.read_point_cloud(filename)
    if not pcd.has_points():
        raise ValueError(f"Point cloud file '{filename}' is empty or invalid.")
    return pcd

def load_transform(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    if len(lines) < 2:
        raise ValueError("Transform file must have at least two lines (translation and quaternion).")
    t = np.array([float(x) for x in lines[0].split()])
    q_vals = [float(x) for x in lines[1].split()]
    qw, qx, qy, qz = q_vals
    R = np.array([
        [1 - 2*(qy**2+qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)]
    ])
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T

def pose_to_transformation(position, quaternion):
    q_w, q_x, q_y, q_z = quaternion
    R = np.array([
        [1-2*(q_y*q_y+q_z*q_z), 2*(q_x*q_y - q_z*q_w), 2*(q_x*q_z+q_y*q_w)],
        [2*(q_x*q_y+q_z*q_w), 1-2*(q_x*q_x+q_z*q_z), 2*(q_y*q_z-q_x*q_w)],
        [2*(q_x*q_z-q_y*q_w), 2*(q_y*q_z+q_x*q_w), 1-2*(q_x*q_x+q_y*q_y)]
    ])
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = position
    return T

def load_box_dimensions(filename):
    """
    Loads the box dimensions from a file saved by the C++ code.
    Expects two lines:
      - First line: min (x y z w)
      - Second line: max (x y z w)
    Returns two numpy arrays (each of shape (4,))
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    if len(lines) < 2:
        raise ValueError("Box dimensions file must have 2 lines")
    min_vals = np.array([float(x) for x in lines[0].split()])
    max_vals = np.array([float(x) for x in lines[1].split()])
    return min_vals, max_vals

def create_cutter_obb(box_min, box_max, T_wp):
    """
    Given the saved box dimensions (box_min and box_max) from the C++ code—which 
    represent the cutter's bounding box in the cutter's local (initial) frame—and 
    a waypoint transform T_wp (4x4), this function computes the oriented bounding box.
    The local box is defined by its minimum and maximum points (using only x,y,z),
    and the OBB is obtained by transforming the box center and using T_wp's rotation.
    """
    # Use only the x,y,z components.
    min_bound = box_min[:3]
    max_bound = box_max[:3]
    
    # Compute the local center and full extents.
    local_center = (min_bound + max_bound) / 2.0
    extent = max_bound - min_bound  # Full sizes along each axis
    
    # Transform the local center to the world frame using T_wp.
    local_center_hom = np.hstack((local_center, 1))
    global_center = (T_wp @ local_center_hom)[:3]
    
    # The orientation is given by the rotation part of T_wp.
    R = T_wp[:3, :3]
    
    # Create and return the oriented bounding box.
    obb = o3d.geometry.OrientedBoundingBox(global_center, R, extent)
    return obb

# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    # ----------------------
    # Load trajectory and related files.
    # ----------------------
    positions, eulers = load_trajectory_euler("trajectory.csv")
    print("Loaded trajectory with", positions.shape[0], "points.")
    quaternions = np.array([eulerZYXToQuaternion(r, p, y) for (r, p, y) in eulers])
    
    try:
        cutter_transform_rel = load_transform("cutter_transform.txt")
        print("Loaded cutter relative transform from file.")
    except Exception as e:
        print("Failed to load cutter relative transform:", e)
        cutter_transform_rel = np.eye(4)
    
    # Load box dimensions for the cutter (saved by C++ after scaling/centering)
    try:
        box_min, box_max = load_box_dimensions("cutter_box.txt")
        print("Loaded cutter box dimensions from file.")
    except Exception as e:
        print("Failed to load cutter box dimensions:", e)
        box_min, box_max = None, None

    # ----------------------
    # Load static objects: goals, obstacles, and point cloud.
    # ----------------------
    static_geometries = []
    start_sphere = create_sphere_at_position(positions[0], radius=0.015, color=(0, 1, 0))
    final_sphere = create_sphere_at_position(positions[-1], radius=0.015, color=(1, 0, 0))
    static_geometries += [start_sphere, final_sphere]

    # Goals.
    goal_spheres = []
    goal_frames  = []
    try:
        goals_pos, goals_eulers = load_goals_euler("goals.csv")
        print("Loaded", goals_pos.shape[0], "goals.")
        for i in range(goals_pos.shape[0]):
            gpos = goals_pos[i]
            geul = goals_eulers[i]
            gquat = eulerZYXToQuaternion(geul[0], geul[1], geul[2])
            sphere = create_sphere_at_position(gpos, radius=0.01, color=(0.0, 0.5, 1.0))
            frame  = create_coordinate_frame(gpos, gquat, scale=0.04)
            goal_spheres.append(sphere)
            goal_frames.append(frame)
        static_geometries += goal_spheres + goal_frames
    except OSError:
        print("No 'goals.csv' file found or unable to read. Skipping goal visualization.")

    # Obstacles.
    obstacle_spheres = []
    try:
        obstacles = load_obstacles("obstacles.csv")
        print(f"Loaded {len(obstacles)} obstacles.")
        for (center, radius) in obstacles:
            sphere = create_sphere_at_position(center, radius=radius, color=(0.0, 0.0, 1.0))
            obstacle_spheres.append(sphere)
        static_geometries += obstacle_spheres
    except OSError:
        print("No 'obstacles.csv' found or unable to read. Skipping obstacle visualization.")

    # Point cloud.
    point_cloud = None
    try:
        point_cloud = load_point_cloud("../data/vine_simple_streo_scan.pcd")
        print(f"Loaded point cloud with {len(point_cloud.points)} points.")
        static_geometries.append(point_cloud)
    except Exception as e:
        print(f"Failed to load point cloud: {e}")
        
    # try:
    #     point_cloud = load_point_cloud("collision_debug.pcd")
    #     print(f"Loaded point cloud with {len(point_cloud.points)} points.")
    #     point_cloud.paint_uniform_color([1, 0, 0]) 
    #     static_geometries.append(point_cloud)
    # except Exception as e:
    #     print(f"Failed to load point cloud: {e}")
        
    # ----------------------
    # Prepare animation objects.
    # ----------------------
    traj_line_set = o3d.geometry.LineSet()
    traj_line_set.points = o3d.utility.Vector3dVector([])
    traj_line_set.lines  = o3d.utility.Vector2iVector([])
    traj_line_set.colors = o3d.utility.Vector3dVector([])

    try:
        cutter_mesh = o3d.io.read_triangle_mesh("../data/cutter.stl")
        cutter_mesh.compute_vertex_normals()
        cutter_scale = 1
        cutter_mesh.scale(cutter_scale, center=cutter_mesh.get_center())
        cutter_mesh.translate(-cutter_mesh.get_center())
    except Exception as e:
        print(f"Failed to load cutter STL: {e}")
        cutter_mesh = None

    cutter_instances = []      # List of cutter meshes (latest opaque, previous ghosted)
    cutter_bbox_instances = [] # Corresponding bounding boxes
    opaque_color = [0.0, 1, 0.0]
    ghost_color  = [0.8, 0.8, 0.8]

    # ----------------------
    # Set up the Open3D Visualizer (legacy).
    # ----------------------
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Trajectory Animation", width=800, height=600)
    for geo in static_geometries:
        vis.add_geometry(geo)
    vis.add_geometry(traj_line_set)

    num_waypoints = positions.shape[0]
    traj_points = []

    for i in range(num_waypoints):
        current_pos = positions[i]
        current_quat = quaternions[i]
        traj_points.append(current_pos)
        traj_line_set.points = o3d.utility.Vector3dVector(traj_points)
        if len(traj_points) >= 2:
            lines = [[j, j+1] for j in range(len(traj_points)-1)]
            traj_line_set.lines = o3d.utility.Vector2iVector(lines)
            colors = [[1, 0, 0] for _ in lines]
            traj_line_set.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(traj_line_set)

        frame = create_coordinate_frame(current_pos, current_quat, scale=0.03)
        vis.add_geometry(frame)

        if cutter_mesh is not None:
            cutter_instance = copy.deepcopy(cutter_mesh)
            # Compute the cutter instance's transform:
            T = pose_to_transformation(current_pos, current_quat)
            T = T @ cutter_transform_rel
            cutter_instance.transform(T)
            cutter_instance.paint_uniform_color(opaque_color)
            vis.add_geometry(cutter_instance)
            cutter_instances.append(cutter_instance)

            # Create a corresponding oriented bounding box for this cutter instance.
            # if box_min is not None and box_max is not None:
            #     T = pose_to_transformation(current_pos, current_quat)
            #     obb = create_cutter_obb(box_min, box_max, T)
            #     obb.color = (0, 1, 0)  # Green edges
            #     vis.add_geometry(obb)
            #     cutter_bbox_instances.append(obb)

            # Update previously added cutter instances and boxes to ghost color.
            for past_cutter in cutter_instances[:-1]:
                past_cutter.paint_uniform_color(ghost_color)
                vis.update_geometry(past_cutter)
            for past_bbox in cutter_bbox_instances[:-1]:
                past_bbox.color = (0.7, 0.7, 0.7)
                vis.update_geometry(past_bbox)

        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.01)

    vis.run()
    vis.destroy_window()
