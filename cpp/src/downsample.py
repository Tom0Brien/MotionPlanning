import open3d as o3d
import sys
import os

def create_mesh_from_point_cloud(pcd, alpha):
    """
    Create a mesh from a point cloud using the alpha shape algorithm.
    
    Parameters:
        pcd (open3d.geometry.PointCloud): The input point cloud.
        alpha (float): The alpha value used for the alpha shape.
        
    Returns:
        open3d.geometry.TriangleMesh: The resulting mesh with computed vertex normals.
    """
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()
    return mesh

def downsample_mesh(mesh, ratio):
    """
    Downsample (simplify) a mesh using quadric edge collapse decimation.
    
    Parameters:
        mesh (open3d.geometry.TriangleMesh): The input mesh.
        ratio (float): A value between 0 and 1 representing the fraction of triangles to keep.
                       For example, 0.5 will attempt to reduce the triangle count by half.
    
    Returns:
        open3d.geometry.TriangleMesh: The simplified mesh.
    """
    original_triangle_count = len(mesh.triangles)
    if not (0 < ratio <= 1):
        print("Error: Downsampling ratio must be a float between 0 (exclusive) and 1 (inclusive).")
        return mesh

    # If ratio is 1.0, no downsampling is performed.
    if ratio == 1.0:
        return mesh

    target_triangles = max(4, int(original_triangle_count * ratio))
    print(f"Downsampling mesh from {original_triangle_count} to approximately {target_triangles} triangles.")
    simplified_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
    simplified_mesh.compute_vertex_normals()
    return simplified_mesh

def point_cloud_to_mesh(input_path, alpha, downsample_ratio=1.0, output_path=None):
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(input_path)
    if not pcd.has_points():
        print("Error: The input file does not contain valid point cloud data.")
        return None

    print(f"Loaded point cloud: {input_path}")
    print(f"Point cloud has {len(pcd.points)} points.")

    # Create a mesh using the alpha shape method
    mesh = create_mesh_from_point_cloud(pcd, alpha)
    print(f"Created mesh with {len(mesh.triangles)} triangles.")

    # Downsample the mesh if requested
    if downsample_ratio < 1.0:
        mesh = downsample_mesh(mesh, downsample_ratio)
        print(f"Downsampled mesh has {len(mesh.triangles)} triangles.")
    else:
        print("No downsampling performed on the mesh.")

    # Determine the output file name if not provided
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}_mesh.obj"

    # Save the mesh as an OBJ file
    success = o3d.io.write_triangle_mesh(output_path, mesh)
    if success:
        print(f"Mesh saved to: {output_path}")
    else:
        print("Error: Could not save the mesh.")

    return mesh

def main():
    if len(sys.argv) < 3:
        print("Usage: python point_cloud_to_mesh_alpha_downsample.py <input_cloud> <alpha> [downsample_ratio] [output_mesh]")
        sys.exit(1)

    input_cloud = sys.argv[1]

    try:
        alpha = float(sys.argv[2])
    except ValueError:
        print("Error: alpha must be a float (e.g., 0.05).")
        sys.exit(1)

    # Set downsample_ratio to 1.0 (no downsampling) if not provided
    if len(sys.argv) >= 4:
        try:
            downsample_ratio = float(sys.argv[3])
        except ValueError:
            print("Error: downsample_ratio must be a float (e.g., 0.5 for 50%).")
            sys.exit(1)
    else:
        downsample_ratio = 1.0

    output_mesh = sys.argv[4] if len(sys.argv) >= 5 else None

    point_cloud_to_mesh(input_cloud, alpha, downsample_ratio, output_mesh)

if __name__ == "__main__":
    main()
