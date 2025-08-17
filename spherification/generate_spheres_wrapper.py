#!/usr/bin/env python3
"""
Wrapper script that replicates generate_sphere_urdf.py functionality
but outputs JSON results for integration with Blender spherification utils.
"""

import argparse
import json
import sys
from pathlib import Path

# Add foam to path
sys.path.insert(0, '/opt/foam')

from foam import SpherizationHelper, load_urdf, get_urdf_meshes, get_urdf_primitives
from trimesh.primitives import Sphere
from trimesh.nsphere import minimum_nsphere

def generate_spheres_for_urdf(urdf_path, mesh_type="collision", depth=1, branch=8, 
                             method="medial", threads=16, shrinkage=1.0,
                             use_volume_heuristic=True, volume_heuristic_ratio=0.7,
                             **foam_kwargs):
    """
    Generate spheres for all meshes in a URDF using the same approach as generate_sphere_urdf.py
    """
    urdf_path = Path(urdf_path)
    urdf_dir = urdf_path.parent
    
    # Set default foam parameters (matching generate_sphere_urdf.py defaults)
    foam_params = {
        'testerLevels': 2,
        'numCover': 5000,
        'minCover': 5,
        'initSpheres': 1000,
        'minSpheres': 200,
        'erFact': 2,
        'expand': True,
        'merge': True,
        'burst': False,
        'optimise': True,
        'maxOptLevel': 1,
        'balExcess': 0.05,
        'verify': True,
        'num_samples': 500,
        'min_samples': 1,
        'manifold_leaves': 1000,
        'simplification_ratio': 0.2,
    }
    foam_params.update(foam_kwargs)
    
    # Create temporary database file
    database_path = urdf_dir / "temp_sphere_database.json"
    
    results = {'collision_spheres': {}, 'self_collision_spheres': {}}
    
    try:
        # Create SpherizationHelper
        sh = SpherizationHelper(database_path, threads)
        
        # Load URDF and get meshes
        urdf = load_urdf(urdf_path)
        meshes = get_urdf_meshes(urdf, shrinkage)
        
        # Filter meshes based on mesh_type preference
        filtered_meshes = []
        for mesh in meshes:
            # mesh.name format is typically "link_name:mesh_type"
            link_name = mesh.name.split(":")[0]
            mesh_name_type = mesh.name.split(":")[1] if ":" in mesh.name else "visual"
            
            # Apply mesh type filtering
            if mesh_type == "collision" and "collision" in mesh_name_type.lower():
                filtered_meshes.append(mesh)
            elif mesh_type == "visual" and "visual" in mesh_name_type.lower():
                filtered_meshes.append(mesh)
            elif mesh_type == "collision" and "visual" in mesh_name_type.lower():
                # Fallback: use visual if no collision available
                print(f"INFO: Using visual mesh for {link_name} (no collision mesh available)", file=sys.stderr)
                filtered_meshes.append(mesh)
            elif mesh_type == "visual" and "collision" in mesh_name_type.lower():
                # Fallback: use collision if no visual available
                print(f"INFO: Using collision mesh for {link_name} (no visual mesh available)", file=sys.stderr)
                filtered_meshes.append(mesh)
        
        # If no filtering worked, use all meshes
        if not filtered_meshes:
            print(f"WARNING: No {mesh_type} meshes found, using all available meshes", file=sys.stderr)
            filtered_meshes = meshes
        
        # Process each mesh
        for mesh in filtered_meshes:
            link_name = mesh.name.split(":")[0]
            
            # Calculate adaptive branch value using volume heuristic
            if use_volume_heuristic:
                center, radius = minimum_nsphere(mesh.mesh.vertices)
                sphere_volume = Sphere(radius, center).volume
                mesh_volume = mesh.mesh.volume
                volume_ratio = sphere_volume / mesh_volume if mesh_volume > 0 else 1.0
                branch_value = min(int(volume_ratio * volume_heuristic_ratio), branch)
                branch_value = max(branch_value, 1)  # Ensure at least 1
            else:
                branch_value = branch
            
            print(f"Processing {mesh.name} with {branch_value} target spheres", file=sys.stderr)
            
            try:
                # Spherize mesh
                sh.spherize_mesh(
                    mesh.name,
                    mesh.mesh,
                    mesh.scale,
                    mesh.xyz,
                    method,
                    mesh.rpy,
                    depth=depth,
                    branch=branch_value,
                    **foam_params
                )
                
                # Get spherization results
                mesh_spheres = sh.get_spherization(mesh.name, depth, branch_value)
                
                if mesh_spheres:
                    # Convert foam spheres to our format
                    spheres_list = []
                    
                    # Handle different possible foam result formats
                    if hasattr(mesh_spheres, 'spheres'):
                        sphere_data_list = mesh_spheres.spheres
                    elif isinstance(mesh_spheres, (list, tuple)):
                        sphere_data_list = mesh_spheres
                    else:
                        sphere_data_list = [mesh_spheres] if mesh_spheres else []
                    
                    for sphere_data in sphere_data_list:
                        if hasattr(sphere_data, 'origin') and hasattr(sphere_data, 'radius'):
                            # foam.model.Sphere has origin (numpy array) and radius attributes
                            origin = sphere_data.origin
                            spheres_list.append({
                                "origin": [float(origin[0]), float(origin[1]), float(origin[2])],
                                "radius": float(sphere_data.radius)
                            })
                        elif isinstance(sphere_data, dict) and 'origin' in sphere_data and 'radius' in sphere_data:
                            spheres_list.append(sphere_data)
                    
                    results['collision_spheres'][link_name] = spheres_list
                    results['self_collision_spheres'][link_name] = spheres_list  # Same for now
                    print(f"Generated {len(spheres_list)} spheres for {link_name}", file=sys.stderr)
                    
                else:
                    print(f"WARNING: No spheres generated for {link_name}, using fallback", file=sys.stderr)
                    fallback_sphere = [{"origin": [0.0, 0.0, 0.0], "radius": 0.05}]
                    results['collision_spheres'][link_name] = fallback_sphere
                    results['self_collision_spheres'][link_name] = fallback_sphere
                    
            except Exception as e:
                print(f"ERROR: Error processing {mesh.name}: {e}", file=sys.stderr)
                fallback_sphere = [{"origin": [0.0, 0.0, 0.0], "radius": 0.05}]
                results['collision_spheres'][link_name] = fallback_sphere
                results['self_collision_spheres'][link_name] = fallback_sphere
        
        # Also process primitives if any
        try:
            primitives = get_urdf_primitives(urdf, shrinkage)
            for primitive in primitives:
                link_name = primitive.name.split(":")[0]
                
                if use_volume_heuristic:
                    center, radius = minimum_nsphere(primitive.mesh.vertices)
                    sphere_volume = Sphere(radius, center).volume
                    mesh_volume = primitive.mesh.volume
                    volume_ratio = sphere_volume / mesh_volume if mesh_volume > 0 else 1.0
                    branch_value = min(int(volume_ratio * volume_heuristic_ratio), branch)
                    branch_value = max(branch_value, 1)
                else:
                    branch_value = branch
                
                print(f"Processing primitive {primitive.name} with {branch_value} target spheres", file=sys.stderr)
                
                sh.spherize_mesh(
                    primitive.name,
                    primitive.mesh,
                    primitive.scale,
                    primitive.xyz,
                    method,
                    primitive.rpy,
                    depth=depth,
                    branch=branch_value,
                    **foam_params
                )
                
                primitive_spheres = sh.get_spherization(primitive.name, depth, branch_value, cache=False)
                if primitive_spheres:
                    spheres_list = []
                    
                    if hasattr(primitive_spheres, 'spheres'):
                        sphere_data_list = primitive_spheres.spheres
                    elif isinstance(primitive_spheres, (list, tuple)):
                        sphere_data_list = primitive_spheres
                    else:
                        sphere_data_list = [primitive_spheres] if primitive_spheres else []
                    
                    for sphere_data in sphere_data_list:
                        if hasattr(sphere_data, 'origin') and hasattr(sphere_data, 'radius'):
                            # foam.model.Sphere has origin (numpy array) and radius attributes
                            origin = sphere_data.origin
                            spheres_list.append({
                                "origin": [float(origin[0]), float(origin[1]), float(origin[2])],
                                "radius": float(sphere_data.radius)
                            })
                        elif isinstance(sphere_data, dict) and 'origin' in sphere_data and 'radius' in sphere_data:
                            spheres_list.append(sphere_data)
                    
                    if link_name in results['collision_spheres']:
                        results['collision_spheres'][link_name].extend(spheres_list)
                        results['self_collision_spheres'][link_name].extend(spheres_list)
                    else:
                        results['collision_spheres'][link_name] = spheres_list
                        results['self_collision_spheres'][link_name] = spheres_list
                    
                    print(f"Added {len(spheres_list)} primitive spheres for {link_name}", file=sys.stderr)
                    
        except Exception as e:
            print(f"INFO: No primitives found or error processing primitives: {e}", file=sys.stderr)
    
    finally:
        # Clean up temporary database file
        if database_path.exists():
            database_path.unlink()
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Generate spheres for URDF using foam")
    parser.add_argument("urdf_path", help="Path to URDF file")
    parser.add_argument("--mesh-type", choices=["visual", "collision"], default="collision",
                       help="Type of mesh to use (default: collision)")
    parser.add_argument("--depth", type=int, default=1, help="Foam depth parameter")
    parser.add_argument("--branch", type=int, default=8, help="Base branch parameter")
    parser.add_argument("--method", default="medial", help="Spherization method")
    parser.add_argument("--threads", type=int, default=16, help="Number of threads")
    parser.add_argument("--shrinkage", type=float, default=1.0, help="Scale factor for meshes")
    parser.add_argument("--no-volume-heuristic", action="store_true", 
                       help="Disable volume-based branch adjustment")
    parser.add_argument("--volume-heuristic-ratio", type=float, default=0.7,
                       help="Ratio for volume heuristic calculation")
    
    args = parser.parse_args()
    
    # Generate spheres
    results = generate_spheres_for_urdf(
        args.urdf_path,
        mesh_type=args.mesh_type,
        depth=args.depth,
        branch=args.branch,
        method=args.method,
        threads=args.threads,
        shrinkage=args.shrinkage,
        use_volume_heuristic=not args.no_volume_heuristic,
        volume_heuristic_ratio=args.volume_heuristic_ratio
    )
    
    # Output results as JSON
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main() 