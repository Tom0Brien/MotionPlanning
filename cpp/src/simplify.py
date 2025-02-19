import open3d as o3d
import sys
import os

def simplify_mesh(input_path, ratio, output_path=None):
    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(input_path)
    if not mesh.has_triangles():
        print("Error: The input file does not contain a valid triangle mesh.")
        return None

    original_triangle_count = len(mesh.triangles)
    print(f"Loaded mesh: {input_path}")
    print(f"Original mesh has {original_triangle_count} triangles.")

    # Validate the ratio value
    if not (0 < ratio < 1):
        print("Error: Ratio must be a float between 0 and 1 (e.g., 0.5 for 50%).")
        return None

    # Compute the target number of triangles
    target_triangles = max(4, int(original_triangle_count * ratio))
    print(f"Target number of triangles: {target_triangles}")

    # Simplify the mesh using quadric edge collapse decimation
    mesh_simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
    print(f"Simplified mesh has {len(mesh_simplified.triangles)} triangles.")

    # Optionally recompute normals for better visual quality
    mesh_simplified.compute_vertex_normals()

    # Determine output file name if not provided
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_simplified{ext}"

    # Save the simplified mesh
    success = o3d.io.write_triangle_mesh(output_path, mesh_simplified)
    if success:
        print(f"Simplified mesh saved to: {output_path}")
    else:
        print("Error: Could not save the simplified mesh.")

    return mesh_simplified

def main():
    if len(sys.argv) < 3:
        print("Usage: python mesh_simplify.py <input_mesh> <ratio> [output_mesh]")
        sys.exit(1)

    input_mesh = sys.argv[1]
    try:
        ratio = float(sys.argv[2])
    except ValueError:
        print("Error: ratio must be a float (e.g., 0.5 for 50%).")
        sys.exit(1)

    output_mesh = sys.argv[3] if len(sys.argv) >= 4 else None

    simplify_mesh(input_mesh, ratio, output_mesh)

if __name__ == "__main__":
    main()
