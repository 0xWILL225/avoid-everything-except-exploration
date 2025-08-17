import bpy
import json
import xml.etree.ElementTree as ET
import os
from pathlib import Path
import subprocess


def add_sphere(x, y, z, radius, index=None):
  """
  Adds a green, opaque sphere to the scene.
  
  Automatically detects the current editing context (link and sphere type) 
  from existing collections and names the sphere appropriately.
  Falls back to manual naming if no context is found.
  """
  # Try to detect current editing context from existing collections
  link_name = None
  sphere_type = None
  sphere_collection = None
  
  # Look for sphere collections to determine context
  for collection in bpy.data.collections:
    if collection.name.endswith('_collision_spheres'):
      link_name = collection.name[:-len('_collision_spheres')]
      sphere_type = 'collision'
      sphere_collection = collection
      break
    elif collection.name.endswith('_self_collision_spheres'):
      link_name = collection.name[:-len('_self_collision_spheres')]
      sphere_type = 'self_collision'
      sphere_collection = collection
      break
  
  # Create the sphere
  bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=(x, y, z))
  obj = bpy.context.active_object
  
  # Name the sphere based on context
  if link_name and sphere_type:
    # Find the next available index for this link/type
    sphere_prefix = f"{link_name}_{sphere_type}_"
    existing_indices = []
    
    for existing_obj in bpy.data.objects:
      if existing_obj.name.startswith(sphere_prefix):
        # Extract index from name like "panda_link1_collision_0"
        try:
          index_str = existing_obj.name[len(sphere_prefix):]
          # Handle names with description like "panda_link1_collision_0_manual"
          if '_' in index_str:
            index_str = index_str.split('_')[0]
          existing_index = int(index_str)
          existing_indices.append(existing_index)
        except ValueError:
          continue
    
    # Get next available index
    next_index = max(existing_indices, default=-1) + 1
    obj.name = f"{sphere_prefix}{next_index}"
    
    # Move to sphere collection
    if move_object_to_collection(obj, sphere_collection):
      print(f"SUCCESS: Added sphere {obj.name} at ({x:.3f}, {y:.3f}, {z:.3f}), r={radius:.3f}")
      print(f"  Auto-detected context: {link_name} {sphere_type}")
    else:
      print(f"WARNING: Added sphere {obj.name} but failed to move to collection properly")
      print(f"  Position: ({x:.3f}, {y:.3f}, {z:.3f}), r={radius:.3f}")
  else:
    # Fall back to manual naming
    name_prefix="sphere_"
    obj.name = f"{name_prefix}{index}" if index is not None else name_prefix
    print(f"WARNING: No link context detected - using manual naming: {obj.name}")
    print(f"  Use load_link_for_editing() first to set proper context")
    print(f"  To store spheres, the correct naming scheme must be {link_name}_{sphere_type}_{index}")

  # Add green material
  mat = bpy.data.materials.new(name="GreenMaterial")
  mat.use_nodes = True
  bsdf = mat.node_tree.nodes.get("Principled BSDF")
  bsdf.inputs["Base Color"].default_value = (0.0, 1.0, 0.0, 1.0)
  bsdf.inputs["Alpha"].default_value = 1.0
  mat.blend_method = 'OPAQUE'
  obj.data.materials.append(mat)

  return obj


def parse_urdf(urdf_path):
  """Parse URDF file and extract link information with mesh paths."""
  tree = ET.parse(urdf_path)
  root = tree.getroot()
  
  links = {}
  urdf_dir = Path(urdf_path).parent
  
  for link in root.findall('link'):
    link_name = link.get('name')
    visual_mesh = None
    collision_mesh = None
    
    # Find visual mesh
    visual = link.find('visual')
    if visual is not None:
      geometry = visual.find('geometry')
      if geometry is not None:
        mesh = geometry.find('mesh')
        if mesh is not None:
          visual_mesh = mesh.get('filename')
    
    # Find collision mesh
    collision = link.find('collision')
    if collision is not None:
      geometry = collision.find('geometry')
      if geometry is not None:
        mesh = geometry.find('mesh')
        if mesh is not None:
          collision_mesh = mesh.get('filename')
    
    links[link_name] = {
      'visual_mesh': visual_mesh,
      'collision_mesh': collision_mesh
    }
  
  return links


def analyze_robot_spheres(urdf_path):
  """
  Step 3: Analyze robot URDF and show table of links with sphere counts.
  """
  print(f"\n=== ROBOT ANALYSIS: {urdf_path} ===")
  
  # Parse URDF
  try:
    links = parse_urdf(urdf_path)
    print(f"SUCCESS: URDF parsed successfully - Found {len(links)} links")
  except Exception as e:
    print(f"ERROR: Error parsing URDF: {e}")
    return None
  
  # Check for sphere files in collision_spheres subdirectory
  urdf_dir = Path(urdf_path).parent
  collision_spheres_dir = urdf_dir / "collision_spheres"
  collision_spheres_path = collision_spheres_dir / "collision_spheres.json"
  self_collision_spheres_path = collision_spheres_dir / "self_collision_spheres.json"
  
  collision_spheres = {}
  self_collision_spheres = {}
  
  if collision_spheres_path.exists():
    with open(collision_spheres_path, 'r') as f:
      collision_spheres = json.load(f)
    print(f"SUCCESS: Found collision_spheres.json")
  else:
    print(f"WARNING: No collision_spheres.json found in {collision_spheres_dir}")
  
  if self_collision_spheres_path.exists():
    with open(self_collision_spheres_path, 'r') as f:
      self_collision_spheres = json.load(f)
    print(f"SUCCESS: Found self_collision_spheres.json")
  else:
    print(f"WARNING: No self_collision_spheres.json found in {collision_spheres_dir}")
  
  # Calculate column widths for consistent formatting
  rows = []
  for link_name, link_info in links.items():
    collision_count = len(collision_spheres.get(link_name, []))
    self_collision_count = len(self_collision_spheres.get(link_name, []))
    has_visual = "Yes" if link_info['visual_mesh'] else "No"
    rows.append([link_name, has_visual, str(collision_count), str(self_collision_count)])
  
  # Calculate maximum width for each column
  headers = ["Link Name", "Visual Mesh", "Collision", "Self-Collision"]
  all_rows = [headers] + rows
  col_widths = []
  for i in range(len(headers)):
    max_width = max(len(row[i]) for row in all_rows)
    col_widths.append(max_width + 2) # Add 2 for padding
  
  # Print formatted table
  header_format = "".join(f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers)))
  print(f"\n{header_format}")
  print("-" * sum(col_widths))
  
  for row in rows:
    row_format = "".join(f"{row[i]:<{col_widths[i]}}" for i in range(len(row)))
    print(row_format)
  
  return {
    'links': links,
    'collision_spheres': collision_spheres,
    'self_collision_spheres': self_collision_spheres,
    'urdf_dir': urdf_dir
  }


def generate_initial_spheres(urdf_path, mesh_type="collision", 
                           depth=1, branch=8, method="medial", 
                           threads=16, shrinkage=1.0,
                           use_volume_heuristic=True, volume_heuristic_ratio=0.7,
                           **foam_kwargs):
  """
  Step 4: Generate initial spheres using foam for meshes in the URDF.
  Uses the same approach as generate_sphere_urdf.py for consistent results.
  
  Args:
    urdf_path: Path to the URDF file
    mesh_type: Type of mesh to use - "visual" (detailed) or "collision" (simplified)
    depth: Foam depth parameter
    branch: Base branch parameter (will be adjusted per mesh using volume heuristic)
    method: Spherization method ("medial", etc.)
    threads: Number of threads for processing
    shrinkage: Scale factor for meshes
    use_volume_heuristic: Whether to use volume-based branch adjustment
    volume_heuristic_ratio: Ratio for volume heuristic calculation
    **foam_kwargs: Additional foam parameters (testerLevels, numCover, etc.)
  """
  print(f"\n=== GENERATING INITIAL SPHERES (using {mesh_type} meshes) ===")
  
  if mesh_type not in ["visual", "collision"]:
    print(f"ERROR: mesh_type must be 'visual' or 'collision', got '{mesh_type}'")
    return False
  
  # Use the wrapper script to avoid Python version compatibility issues
  wrapper_script = Path(__file__).parent / "generate_spheres_wrapper.py"
  
  # Build command arguments
  cmd_args = [
    'python3', str(wrapper_script), str(urdf_path),
    '--mesh-type', mesh_type,
    '--depth', str(depth),
    '--branch', str(branch),
    '--method', method,
    '--threads', str(threads),
    '--shrinkage', str(shrinkage),
    '--volume-heuristic-ratio', str(volume_heuristic_ratio)
  ]
  
  if not use_volume_heuristic:
    cmd_args.append('--no-volume-heuristic')
  
  print(f"Running foam wrapper: {' '.join(cmd_args)}")
  
  try:
    # Run the wrapper script
    result = subprocess.run(cmd_args, capture_output=True, text=True, timeout=300)
    
    if result.returncode != 0:
      print(f"ERROR: Foam wrapper failed with return code {result.returncode}")
      print(f"STDERR: {result.stderr}")
      return False
    
    # Parse JSON output
    try:
      sphere_results = json.loads(result.stdout)
      collision_spheres = sphere_results.get('collision_spheres', {})
      self_collision_spheres = sphere_results.get('self_collision_spheres', {})
      
      print(f"Successfully generated spheres for {len(collision_spheres)} links")
      
    except json.JSONDecodeError as e:
      print(f"ERROR: Failed to parse wrapper output as JSON: {e}")
      print(f"Raw output: {result.stdout[:500]}...")
      return False
    
  except subprocess.TimeoutExpired:
    print(f"ERROR: Foam wrapper timed out after 300 seconds")
    return False
  except Exception as e:
    print(f"ERROR: Error running foam wrapper: {e}")
    return False
  
  # Save generated spheres to collision_spheres subdirectory
  urdf_path = Path(urdf_path)
  urdf_dir = urdf_path.parent
  collision_spheres_dir = urdf_dir / "collision_spheres"
  collision_spheres_dir.mkdir(exist_ok=True)
  
  collision_path = collision_spheres_dir / "collision_spheres.json"
  self_collision_path = collision_spheres_dir / "self_collision_spheres.json"
  
  with open(collision_path, 'w') as f:
    json.dump(collision_spheres, f, indent=2)
  
  with open(self_collision_path, 'w') as f:
    json.dump(self_collision_spheres, f, indent=2)
  
  print(f"SUCCESS: Saved collision_spheres.json and self_collision_spheres.json to {collision_spheres_dir}")
  
  # Re-run analysis to show updated table
  analyze_robot_spheres(urdf_path)
  return True


def parse_foam_output(output):
  """Parse foam output to extract sphere data from JSON format."""
  try:
    import json
    foam_data = json.loads(output)
    
    # Foam outputs an array of results, we want the best one (lowest score)
    if not foam_data:
      print("WARNING: Empty foam output")
      return [{"origin": [0.0, 0.0, 0.0], "radius": 0.05}]
    
    # Find the result with the best (lowest) score
    best_result = min(foam_data, key=lambda x: x.get('best', float('inf')))
    
    # Extract just the spheres, ignoring mean/best/worst scores
    spheres = best_result.get('spheres', [])
    
    print(f"  Foam result: best={best_result.get('best', 'N/A')}, spheres={len(spheres)}")
    
    return spheres
    
  except json.JSONDecodeError as e:
    print(f"WARNING: Failed to parse foam JSON output: {e}")
    print(f"  Raw output: {output[:200]}...")
    return [{"origin": [0.0, 0.0, 0.0], "radius": 0.05}]
  except Exception as e:
    print(f"WARNING: Error processing foam output: {e}")
    return [{"origin": [0.0, 0.0, 0.0], "radius": 0.05}]


def load_link_for_editing(urdf_path, link_name, sphere_type="collision", apply_coordinate_fix=True):
  """
  Step 5: Load a specific link's mesh and spheres into Blender for editing.
  sphere_type: "collision" or "self_collision"
  apply_coordinate_fix: Apply coordinate system transformation for .obj files (Y-up to Z-up)
  """
  print(f"\n=== LOADING {link_name} FOR {sphere_type.upper()} EDITING ===")
  
  # Clear existing objects
  clear_scene()
  
  # Clean up any orphaned spheres that might exist
  clean_orphaned_spheres(link_name, sphere_type)
  
  # Create collections for organization
  mesh_collection, sphere_collection = create_collections(link_name, sphere_type)
  print(f"Created collections: '{mesh_collection.name}' and '{sphere_collection.name}'")
  
  urdf_dir = Path(urdf_path).parent
  
  # Parse URDF to get mesh path
  try:
    links = parse_urdf(urdf_path)
    link_info = links.get(link_name)
    if not link_info:
      print(f"ERROR: Link '{link_name}' not found in URDF")
      return False
  except Exception as e:
    print(f"ERROR: Error parsing URDF: {e}")
    return False
  
  # Load mesh
  mesh_path = None
  if link_info['visual_mesh']:
    mesh_path = urdf_dir / link_info['visual_mesh']
  elif link_info['collision_mesh']:
    mesh_path = urdf_dir / link_info['collision_mesh']
  
  if mesh_path and mesh_path.exists():
    try:
      # Import mesh based on file type
      if mesh_path.suffix.lower() == '.obj':
        bpy.ops.wm.obj_import(filepath=str(mesh_path))
      elif mesh_path.suffix.lower() == '.stl':
        bpy.ops.wm.stl_import(filepath=str(mesh_path))
      elif mesh_path.suffix.lower() == '.dae':
        bpy.ops.wm.collada_import(filepath=str(mesh_path))
      elif mesh_path.suffix.lower() == '.ply':
        bpy.ops.wm.ply_import(filepath=str(mesh_path))
      else:
        print(f"ERROR: Unsupported mesh format: {mesh_path.suffix}")
        print(f"  Supported formats: .obj, .stl, .dae, .ply")
        return False
      
      # Get all imported mesh objects (some .obj files contain multiple meshes)
      imported_objects = bpy.context.selected_objects[:] if bpy.context.selected_objects else []
      
      if imported_objects:
        print(f"Imported {len(imported_objects)} objects from {mesh_path.name}")
        
        # Apply coordinate system fix to ALL imported objects
        if apply_coordinate_fix and mesh_path.suffix.lower() == '.obj':
          for obj in imported_objects:
            if obj.type == 'MESH':
              apply_obj_coordinate_fix(obj)
        
        # Rename and lock all mesh objects
        main_mesh = None
        for i, obj in enumerate(imported_objects):
          if obj.type == 'MESH':
            if i == 0:
              # First mesh gets the main name
              obj.name = f"{link_name}_mesh"
              main_mesh = obj
            else:
              # Additional meshes get numbered names
              obj.name = f"{link_name}_mesh_{i}"
            
            # Lock the mesh object
            obj.lock_location = (True, True, True)
            obj.lock_rotation = (True, True, True)
            obj.lock_scale = (True, True, True)
            
            # Add gray material to mesh
            add_mesh_material(obj)
            
            # Move to mesh collection
            move_object_to_collection(obj, mesh_collection)
            
            print(f" SUCCESS: Loaded and locked: {obj.name}")
        
        print(f"SUCCESS: Loaded and processed {len(imported_objects)} mesh objects")
      else:
        print(f"WARNING: Mesh imported but no objects selected")
      
    except Exception as e:
      print(f"ERROR: Error loading mesh: {e}")
      print(f"  Mesh path: {mesh_path}")
      print(f"  Trying alternative import methods...")
      
      # Try legacy import operators as fallback
      try:
        if mesh_path.suffix.lower() == '.obj':
          bpy.ops.import_scene.obj(filepath=str(mesh_path))
        elif mesh_path.suffix.lower() == '.stl':
          bpy.ops.import_mesh.stl(filepath=str(mesh_path))
        
        # Apply coordinate fix for legacy import too - handle multiple meshes
        imported_objects = bpy.context.selected_objects[:] if bpy.context.selected_objects else []
        
        if imported_objects:
          print(f"Legacy import: {len(imported_objects)} objects from {mesh_path.name}")
          
          # Apply coordinate fix to ALL imported objects
          if apply_coordinate_fix and mesh_path.suffix.lower() == '.obj':
            for obj in imported_objects:
              if obj.type == 'MESH':
                apply_obj_coordinate_fix(obj)
          
          # Rename and lock all mesh objects
          for i, obj in enumerate(imported_objects):
            if obj.type == 'MESH':
              if i == 0:
                obj.name = f"{link_name}_mesh"
              else:
                obj.name = f"{link_name}_mesh_{i}"
              
              # Lock the mesh object
              obj.lock_location = (True, True, True)
              obj.lock_rotation = (True, True, True)
              obj.lock_scale = (True, True, True)
              
              # Add gray material to mesh
              add_mesh_material(obj)
              
              # Move to mesh collection
              move_object_to_collection(obj, mesh_collection)
              
              print(f" SUCCESS: Legacy loaded and locked: {obj.name}")
          
          print(f"SUCCESS: Legacy import processed {len(imported_objects)} mesh objects")
        else:
          print(f"WARNING: Legacy import succeeded but no objects found")
        
      except Exception as e2:
        print(f"ERROR: Legacy import also failed: {e2}")
        print(f"  Continuing without mesh...")
  else:
    print(f"WARNING: No mesh found for link '{link_name}'")
  
  # Load spheres from collision_spheres subdirectory
  sphere_file = f"{sphere_type}_spheres.json"
  collision_spheres_dir = urdf_dir / "collision_spheres"
  sphere_path = collision_spheres_dir / sphere_file
  
  if sphere_path.exists():
    with open(sphere_path, 'r') as f:
      sphere_data = json.load(f)
    
    link_spheres = sphere_data.get(link_name, [])
    
    for i, sphere in enumerate(link_spheres):
      origin = sphere['origin']
      radius = sphere['radius']
      sphere_obj = add_sphere(origin[0], origin[1], origin[2], radius)
      # Override auto-detected name for loading from file
      sphere_obj.name = f"{link_name}_{sphere_type}_{i}"
      
      # Move sphere to sphere collection
      move_object_to_collection(sphere_obj, sphere_collection)
      
      print(f" SUCCESS: Added sphere {i}: pos=({origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}), r={radius:.3f}")
    
    print(f"SUCCESS: Loaded {len(link_spheres)} {sphere_type} spheres for {link_name}")
    return True
  else:
    print(f"ERROR: No {sphere_file} found")
    return False



def clear_scene():
  """Clear all objects from the Blender scene and remove link-related collections."""
  # Remove all objects first
  bpy.ops.object.select_all(action='SELECT')
  bpy.ops.object.delete(use_global=False)
  
  # Remove collections that match our naming patterns
  collections_to_remove = []
  
  for collection in bpy.data.collections:
    # Check if collection matches our naming patterns
    if (collection.name.endswith('_meshes') or 
      collection.name.endswith('_collision_spheres') or 
      collection.name.endswith('_self_collision_spheres')):
      collections_to_remove.append(collection)
  
  # Remove the collections
  for collection in collections_to_remove:
    # Store the name before removing (becomes invalid after removal)
    collection_name = collection.name
    
    # Remove from scene hierarchy first
    if collection_name in bpy.context.scene.collection.children:
      bpy.context.scene.collection.children.unlink(collection)
    
    # Remove from blend data
    bpy.data.collections.remove(collection)
    print(f"CLEANUP: Removed collection: {collection_name}")
  
  if collections_to_remove:
    print(f"SUCCESS: Cleaned up {len(collections_to_remove)} collections")


def add_mesh_material(mesh_obj):
  """Add a gray material to the mesh object."""
  mat = bpy.data.materials.new(name="MeshMaterial")
  mat.use_nodes = True
  bsdf = mat.node_tree.nodes.get("Principled BSDF")
  bsdf.inputs["Base Color"].default_value = (0.7, 0.7, 0.7, 1.0) # Gray
  bsdf.inputs["Alpha"].default_value = 0.7 # Semi-transparent
  mat.blend_method = 'BLEND'
  mesh_obj.data.materials.append(mat)


def save_edited_spheres(urdf_path, link_name, sphere_type="collision"):
  """
  Step 7: Save edited spheres back to JSON file.
  """
  print(f"\n=== SAVING {sphere_type.upper()} SPHERES FOR {link_name} ===")
  
  urdf_dir = Path(urdf_path).parent
  collision_spheres_dir = urdf_dir / "collision_spheres"
  collision_spheres_dir.mkdir(exist_ok=True)  # Create directory if it doesn't exist
  
  sphere_file = f"{sphere_type}_spheres.json"
  sphere_path = collision_spheres_dir / sphere_file
  
  # Extract sphere data from Blender
  spheres = []
  sphere_prefix = f"{link_name}_{sphere_type}_"
  
  for obj in bpy.data.objects:
    if obj.type == 'MESH' and obj.name.startswith(sphere_prefix):
      loc = obj.location
      radius = obj.dimensions.x / 2 # assumes uniform scaling
      spheres.append({
        "origin": [round(loc.x, 6), round(loc.y, 6), round(loc.z, 6)],
        "radius": round(radius, 6)
      })
  
  print(f"Extracted {len(spheres)} spheres from Blender scene")
  
  # Load existing data or create new
  if sphere_path.exists():
    with open(sphere_path, 'r') as f:
      sphere_data = json.load(f)
  else:
    sphere_data = {}
  
  # Update data for this link
  sphere_data[link_name] = spheres
  
  # Save back to file
  with open(sphere_path, 'w') as f:
    json.dump(sphere_data, f, indent=2)
  
  print(f"SUCCESS: Saved {len(spheres)} spheres to {sphere_file}")
  
  # Show summary
  for i, sphere in enumerate(spheres):
    origin = sphere['origin']
    radius = sphere['radius']
    print(f" Sphere {i}: pos=({origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}), r={radius:.3f}")


def apply_obj_coordinate_fix(mesh_obj):
  """
  Apply coordinate system transformation for .obj files.
  Common issue: .obj files use Y-up, but URDF/robotics uses Z-up.
  This applies a -90° rotation around X-axis to convert Y-up to Z-up.
  """
  import mathutils
  
  print(f"TOGGLE: Applying coordinate system fix for {mesh_obj.name}")
  
  # Ensure object is selected and active
  bpy.context.view_layer.objects.active = mesh_obj
  mesh_obj.select_set(True)
  
  # Apply transformation: -90° rotation around X-axis (Y-up to Z-up)
  # This is the most common transformation needed for .obj files in robotics
  rotation_matrix = mathutils.Matrix.Rotation(-1.5707963267948966, 4, 'X') # -90° in radians
  mesh_obj.matrix_world = rotation_matrix @ mesh_obj.matrix_world
  
  # Apply the transformation to make it permanent
  bpy.context.view_layer.update()
  
  print(f"  SUCCESS: Applied Y-up to Z-up transformation")


def apply_custom_mesh_transform(mesh_obj, rotation_axis='X', rotation_degrees=-90):
  """
  Apply custom coordinate system transformation to a mesh object.
  
  Args:
    mesh_obj: Blender mesh object
    rotation_axis: 'X', 'Y', or 'Z' 
    rotation_degrees: Rotation in degrees (e.g., -90, 90, 180)
  """
  import mathutils
  import math
  
  print(f"TOGGLE: Applying custom transform: {rotation_degrees}° around {rotation_axis}-axis")
  
  # Ensure object is selected and active
  bpy.context.view_layer.objects.active = mesh_obj
  mesh_obj.select_set(True)
  
  # Convert degrees to radians
  rotation_radians = math.radians(rotation_degrees)
  
  # Apply transformation
  rotation_matrix = mathutils.Matrix.Rotation(rotation_radians, 4, rotation_axis)
  mesh_obj.matrix_world = rotation_matrix @ mesh_obj.matrix_world
  
  # Apply the transformation to make it permanent
  bpy.context.view_layer.update()
  
  print(f"  SUCCESS: Applied {rotation_degrees}° {rotation_axis}-axis rotation")


def get_mesh_object(link_name):
  """Get the mesh object for a loaded link."""
  mesh_name = f"{link_name}_mesh"
  return bpy.data.objects.get(mesh_name)


def compare_file_formats(urdf_path, link_name):
  """
  If multiple formats exist for the same link, compare their import behavior.
  """
  print(f"\nCOMPARING FILE FORMATS FOR {link_name}")
  print("=" * 50)
  
  urdf_dir = Path(urdf_path).parent
  
  # Look for different format versions of the same mesh
  possible_formats = ['.obj', '.stl', '.dae', '.ply']
  found_meshes = []
  
  # Check visual and collision mesh directories
  for mesh_dir in ['meshes/visual', 'meshes/collision']:
    full_dir = urdf_dir / mesh_dir
    if full_dir.exists():
      for format_ext in possible_formats:
        # Look for files that might be for this link
        pattern = f"*{link_name.replace('panda_', '')}*{format_ext}"
        matching_files = list(full_dir.glob(pattern))
        for file in matching_files:
          found_meshes.append(file)
  
  if found_meshes:
    print("Found alternative mesh formats:")
    for mesh_file in found_meshes:
      print(f"  - {mesh_file}")
    
    print("\nYou can test these different formats to see")
    print("  which ones import with correct orientation!")
  else:
    print("Only one mesh format found for this link")


def load_link_without_coordinate_fix(urdf_path, link_name, sphere_type="collision"):
  """
  Load link exactly as Blender imports it, without any coordinate fixes.
  Useful for debugging the raw import behavior.
  """
  print(f"\nLOADING {link_name} WITHOUT COORDINATE FIXES")
  print("(For debugging import behavior)")
  
  return load_link_for_editing(urdf_path, link_name, sphere_type, apply_coordinate_fix=False)


def print_sphere_info_for_link(link_name, sphere_type="collision"):
  """Print info for all spheres belonging to a specific link and sphere type."""
  sphere_prefix = f"{link_name}_{sphere_type}_"
  found_spheres = []
  
  for obj in bpy.data.objects:
    if obj.type == 'MESH' and obj.name.startswith(sphere_prefix):
      found_spheres.append(obj)
  
  if not found_spheres:
    print(f"ERROR: No spheres found with prefix '{sphere_prefix}'")
    return
  
  print(f"\nSPHERE INFO FOR {link_name} ({sphere_type.upper()})")
  print("=" * 50)
  for obj in found_spheres:
    loc = obj.location
    radius = obj.dimensions.x / 2 # assumes uniform scaling
    print(f" {obj.name}: pos=({loc.x:.3f}, {loc.y:.3f}, {loc.z:.3f}), r={radius:.3f}")


def print_all_current_spheres():
  """Print info for all sphere objects currently in the scene."""
  sphere_objects = []
  
  # Look for objects that might be spheres based on common naming patterns
  for obj in bpy.data.objects:
    if obj.type == 'MESH':
      # Check for sphere-like names
      obj_name_lower = obj.name.lower()
      if any(keyword in obj_name_lower for keyword in ['sphere', 'collision', 'self_collision']):
        sphere_objects.append(obj)
  
  if not sphere_objects:
    print(f"ERROR: No sphere objects found in scene")
    return
  
  print(f"\nALL SPHERE OBJECTS IN SCENE ({len(sphere_objects)} found)")
  print("=" * 60)
  for obj in sphere_objects:
    loc = obj.location
    radius = obj.dimensions.x / 2 # assumes uniform scaling
    print(f" {obj.name}: pos=({loc.x:.3f}, {loc.y:.3f}, {loc.z:.3f}), r={radius:.3f}")


def add_manual_sphere(link_name, sphere_type, x, y, z, radius, description="manual"):
  """
  Add a sphere that will be included when saving spheres for a link.
  
  Args:
    link_name: The robot link name (e.g., 'panda_link1')
    sphere_type: 'collision' or 'self_collision'
    x, y, z: Position coordinates
    radius: Sphere radius
    description: Optional description for the sphere name
  
  Returns:
    The created sphere object
  """
  # Find the next available index for this link/type
  sphere_prefix = f"{link_name}_{sphere_type}_"
  existing_indices = []
  
  for obj in bpy.data.objects:
    if obj.name.startswith(sphere_prefix):
      # Extract index from name like "panda_link1_collision_0"
      try:
        index_str = obj.name[len(sphere_prefix):]
        # Handle names with description like "panda_link1_collision_0_manual"
        if '_' in index_str:
          index_str = index_str.split('_')[0]
        index = int(index_str)
        existing_indices.append(index)
      except ValueError:
        continue
  
  # Get next available index
  next_index = max(existing_indices, default=-1) + 1
  
  # Create sphere with proper naming
  sphere_name = f"{sphere_prefix}{next_index}"
  if description and description != "manual":
    sphere_name += f"_{description}"
  
  sphere_obj = add_sphere(x, y, z, radius)
  sphere_obj.name = sphere_name
  
  print(f"SUCCESS: Added manual sphere: {sphere_name}")
  print(f"  Position: ({x:.3f}, {y:.3f}, {z:.3f}), Radius: {radius:.3f}")
  print(f"  This sphere will be saved when you call save_edited_spheres()")
  
  return sphere_obj


def get_save_naming_requirements(link_name, sphere_type):
  """
  Show the naming requirements for manual spheres to be included in saves.
  """
  sphere_prefix = f"{link_name}_{sphere_type}_"
  
  print(f"\nMANUAL SPHERE NAMING REQUIREMENTS")
  print("=" * 50)
  print(f"Link: {link_name}")
  print(f"Sphere type: {sphere_type}")
  print(f"Required prefix: '{sphere_prefix}'")
  print()
  print("NAMING EXAMPLES:")
  print(f" SUCCESS: {sphere_prefix}0")
  print(f" SUCCESS: {sphere_prefix}1") 
  print(f" SUCCESS: {sphere_prefix}5_custom")
  print(f" SUCCESS: {sphere_prefix}10_manual")
  print(f" ERROR: custom_sphere (wrong prefix)")
  print(f" ERROR: {link_name}_sphere_0 (wrong format)")
  print()
  print("EASY WAY TO ADD MANUAL SPHERES:")
  print(f"  spherification_utils.add_manual_sphere('{link_name}', '{sphere_type}', x, y, z, radius)")
  print()
  print("WHAT GETS SAVED:")
  print(f"  - All objects with names starting with '{sphere_prefix}'")
  print(f"  - Object type must be 'MESH'")
  print(f"  - Radius calculated from object dimensions (X-axis)")


def count_spheres_for_link(link_name, sphere_type="collision"):
  """Count how many spheres exist for a given link and sphere type."""
  sphere_prefix = f"{link_name}_{sphere_type}_"
  count = 0
  
  for obj in bpy.data.objects:
    if obj.type == 'MESH' and obj.name.startswith(sphere_prefix):
      count += 1
  
  return count


def generate_and_load_spheres(urdf_path, link_name, sphere_type="collision", clear_existing=True):
  """
  Generate spheres for a specific link using foam and load them into Blender immediately.
  This replaces existing spheres in the scene if clear_existing=True.
  
  Args:
    urdf_path: Path to URDF file
    link_name: Name of the link to generate spheres for
    sphere_type: "collision" or "self_collision" 
    clear_existing: Whether to clear existing spheres for this link first
  """
  print(f"\n=== GENERATING AND LOADING SPHERES FOR {link_name} ===")
  
  # Parse URDF to get mesh path
  try:
    links = parse_urdf(urdf_path)
    link_info = links.get(link_name)
    if not link_info:
      print(f"ERROR: Link '{link_name}' not found in URDF")
      return False
  except Exception as e:
    print(f"ERROR: Error parsing URDF: {e}")
    return False
  
  urdf_dir = Path(urdf_path).parent
  
  # Get mesh path
  mesh_path = None
  if link_info['visual_mesh']:
    mesh_path = urdf_dir / link_info['visual_mesh']
  elif link_info['collision_mesh']:
    mesh_path = urdf_dir / link_info['collision_mesh']
  
  if not mesh_path or not mesh_path.exists():
    print(f"ERROR: No mesh found for link '{link_name}'")
    return False
  
  print(f"Generating spheres for mesh: {mesh_path}")
  
  # Clear existing spheres for this link if requested
  if clear_existing:
    sphere_prefix = f"{link_name}_{sphere_type}_"
    removed_count = 0
    
    # Get list of objects to remove (can't modify collection while iterating)
    objects_to_remove = []
    for obj in bpy.data.objects:
      if obj.type == 'MESH' and obj.name.startswith(sphere_prefix):
        objects_to_remove.append(obj)
    
    # Remove them
    for obj in objects_to_remove:
      bpy.data.objects.remove(obj, do_unlink=True)
      removed_count += 1
    
    if removed_count > 0:
      print(f"CLEANUP: Removed {removed_count} existing spheres")
  
  # Generate spheres using foam
  generated_spheres = []
  try:
    # Run foam sphere generation
    result = subprocess.run([
      'python3', '/opt/foam/scripts/generate_spheres.py',
      str(mesh_path)
    ], capture_output=True, text=True, timeout=30)
    
    if result.returncode == 0:
      # Parse foam output to extract sphere data
      generated_spheres = parse_foam_output(result.stdout)
      print(f"SUCCESS: Foam generated {len(generated_spheres)} spheres")
    else:
      print(f"ERROR: Foam failed: {result.stderr}")
      # Generate fallback spheres
      generated_spheres = [
        {"origin": [0.0, 0.0, 0.0], "radius": 0.06 if sphere_type == "collision" else 0.1}
      ]
      print(f"WARNING: Using fallback sphere")
      
  except Exception as e:
    print(f"ERROR: Error running foam: {e}")
    # Generate fallback spheres
    generated_spheres = [
      {"origin": [0.0, 0.0, 0.0], "radius": 0.06 if sphere_type == "collision" else 0.1}
    ]
    print(f"WARNING: Using fallback sphere")
  
  # Create collections for organization
  mesh_collection, sphere_collection = create_collections(link_name, sphere_type)
  
  # Load generated spheres into Blender
  for i, sphere in enumerate(generated_spheres):
    origin = sphere['origin']
    radius = sphere['radius']
    sphere_obj = add_sphere(origin[0], origin[1], origin[2], radius)
    # Override auto-detected name for generated spheres
    sphere_obj.name = f"{link_name}_{sphere_type}_{i}"
    
    # Move sphere to sphere collection
    move_object_to_collection(sphere_obj, sphere_collection)
    
    print(f" SUCCESS: Added sphere {i}: pos=({origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}), r={radius:.3f}")
  
  # Save to JSON file in collision_spheres subdirectory
  collision_spheres_dir = urdf_dir / "collision_spheres"
  collision_spheres_dir.mkdir(exist_ok=True)  # Create directory if it doesn't exist
  
  sphere_file = f"{sphere_type}_spheres.json"
  sphere_path = collision_spheres_dir / sphere_file
  
  # Load existing data or create new
  if sphere_path.exists():
    with open(sphere_path, 'r') as f:
      sphere_data = json.load(f)
  else:
    sphere_data = {}
  
  # Update data for this link
  sphere_data[link_name] = generated_spheres
  
  # Save back to file
  with open(sphere_path, 'w') as f:
    json.dump(sphere_data, f, indent=2)
  
  print(f"Saved {len(generated_spheres)} spheres to {collision_spheres_dir}/{sphere_file}")
  print(f"SUCCESS: Generated and loaded {len(generated_spheres)} spheres for {link_name}")
  
  return True


def create_collections(link_name, sphere_type):
  """
  Create or get collections for organizing meshes and spheres.
  
  Returns:
    tuple: (mesh_collection, sphere_collection)
  """
  # Get or create mesh collection
  mesh_collection_name = f"{link_name}_meshes"
  if mesh_collection_name in bpy.data.collections:
    mesh_collection = bpy.data.collections[mesh_collection_name]
  else:
    mesh_collection = bpy.data.collections.new(mesh_collection_name)
    bpy.context.scene.collection.children.link(mesh_collection)
  
  # Get or create sphere collection
  sphere_collection_name = f"{link_name}_{sphere_type}_spheres"
  if sphere_collection_name in bpy.data.collections:
    sphere_collection = bpy.data.collections[sphere_collection_name]
  else:
    sphere_collection = bpy.data.collections.new(sphere_collection_name)
    bpy.context.scene.collection.children.link(sphere_collection)
  
  return mesh_collection, sphere_collection


def move_object_to_collection(obj, target_collection):
  """Move an object to a specific collection, removing it from others."""
  return move_object_to_collection_safe(obj, target_collection)


def hide_meshes(link_name):
  """Hide all mesh objects for a link."""
  mesh_collection_name = f"{link_name}_meshes"
  if mesh_collection_name in bpy.data.collections:
    collection = bpy.data.collections[mesh_collection_name]
    collection.hide_viewport = True
    print(f"HIDDEN: Hidden meshes for {link_name}")
  else:
    print(f"ERROR: No mesh collection found for {link_name}")


def show_meshes(link_name):
  """Show all mesh objects for a link."""
  mesh_collection_name = f"{link_name}_meshes"
  if mesh_collection_name in bpy.data.collections:
    collection = bpy.data.collections[mesh_collection_name]
    collection.hide_viewport = False
    print(f"SHOWN: Shown meshes for {link_name}")
  else:
    print(f"ERROR: No mesh collection found for {link_name}")


def hide_spheres(link_name, sphere_type="collision"):
  """Hide all sphere objects for a link."""
  sphere_collection_name = f"{link_name}_{sphere_type}_spheres"
  if sphere_collection_name in bpy.data.collections:
    collection = bpy.data.collections[sphere_collection_name]
    collection.hide_viewport = True
    print(f"HIDDEN: Hidden {sphere_type} spheres for {link_name}")
  else:
    print(f"ERROR: No sphere collection found for {link_name}")


def show_spheres(link_name, sphere_type="collision"):
  """Show all sphere objects for a link."""
  sphere_collection_name = f"{link_name}_{sphere_type}_spheres"
  if sphere_collection_name in bpy.data.collections:
    collection = bpy.data.collections[sphere_collection_name]
    collection.hide_viewport = False
    print(f"SHOWN: Shown {sphere_type} spheres for {link_name}")
  else:
    print(f"ERROR: No sphere collection found for {link_name}")


def toggle_mesh_visibility(link_name):
  """Toggle visibility of all meshes for a link."""
  mesh_collection_name = f"{link_name}_meshes"
  if mesh_collection_name in bpy.data.collections:
    collection = bpy.data.collections[mesh_collection_name]
    collection.hide_viewport = not collection.hide_viewport
    status = "hidden" if collection.hide_viewport else "shown"
    print(f"TOGGLE: Toggled meshes for {link_name}: {status}")
  else:
    print(f"ERROR: No mesh collection found for {link_name}")


def toggle_sphere_visibility(link_name, sphere_type="collision"):
  """Toggle visibility of all spheres for a link."""
  sphere_collection_name = f"{link_name}_{sphere_type}_spheres"
  if sphere_collection_name in bpy.data.collections:
    collection = bpy.data.collections[sphere_collection_name]
    collection.hide_viewport = not collection.hide_viewport
    status = "hidden" if collection.hide_viewport else "shown"
    print(f"TOGGLE: Toggled {sphere_type} spheres for {link_name}: {status}")
  else:
    print(f"ERROR: No sphere collection found for {link_name}")


def organize_scene_collections(link_name, sphere_type="collision"):
  """
  Organize current scene objects into collections.
  Useful if you loaded objects before collection management was added.
  """
  print(f"\nORGANIZING SCENE INTO COLLECTIONS")
  print("=" * 50)
  
  # Create collections
  mesh_collection, sphere_collection = create_collections(link_name, sphere_type)
  
  # Move meshes to mesh collection
  mesh_count = 0
  for obj in bpy.data.objects:
    if obj.type == 'MESH' and obj.name.startswith(f"{link_name}_mesh"):
      move_object_to_collection(obj, mesh_collection)
      mesh_count += 1
  
  # Move spheres to sphere collection
  sphere_count = 0
  sphere_prefix = f"{link_name}_{sphere_type}_"
  for obj in bpy.data.objects:
    if obj.type == 'MESH' and obj.name.startswith(sphere_prefix):
      move_object_to_collection(obj, sphere_collection)
      sphere_count += 1
  
  print(f"SUCCESS: Organized {mesh_count} meshes into '{mesh_collection.name}' collection")
  print(f"SUCCESS: Organized {sphere_count} spheres into '{sphere_collection.name}' collection")
  
  return mesh_collection, sphere_collection


def cleanup_all_link_collections():
  """
  Manually clean up all link-related collections from the scene.
  Useful if you want to clean up without loading a new link.
  """
  print(f"\nCLEANUP: CLEANING UP ALL LINK COLLECTIONS")
  print("=" * 50)
  
  collections_to_remove = []
  
  for collection in bpy.data.collections:
    # Check if collection matches our naming patterns
    if (collection.name.endswith('_meshes') or 
      collection.name.endswith('_collision_spheres') or 
      collection.name.endswith('_self_collision_spheres')):
      collections_to_remove.append(collection)
  
  if not collections_to_remove:
    print(f"SUCCESS: No link collections found to clean up")
    return
  
  # Remove the collections
  for collection in collections_to_remove:
    # Store the name before removing (becomes invalid after removal)
    collection_name = collection.name
    
    # Remove from scene hierarchy first
    if collection_name in bpy.context.scene.collection.children:
      bpy.context.scene.collection.children.unlink(collection)
    
    # Remove from blend data
    bpy.data.collections.remove(collection)
    print(f"CLEANUP: Removed collection: {collection_name}")
  
  print(f"SUCCESS: Cleaned up {len(collections_to_remove)} collections")


def list_all_collections():
  """List all collections in the current scene."""
  print(f"\nALL COLLECTIONS IN SCENE")
  print("=" * 40)
  
  if not bpy.data.collections:
    print("No collections found")
    return
  
  for collection in bpy.data.collections:
    object_count = len(collection.objects)
    hidden = f"HIDDEN:" if collection.hide_viewport else f"SHOWN:"
    print(f"{hidden} {collection.name} ({object_count} objects)")


def cleanup_empty_collections():
  """Remove all empty collections from the scene."""
  print(f"\nCLEANUP: CLEANING UP EMPTY COLLECTIONS")
  print("=" * 40)
  
  collections_to_remove = []
  
  for collection in bpy.data.collections:
    if len(collection.objects) == 0:
      collections_to_remove.append(collection)
  
  if not collections_to_remove:
    print(f"SUCCESS: No empty collections found")
    return
  
  # Remove the collections
  for collection in collections_to_remove:
    # Store the name before removing (becomes invalid after removal)
    collection_name = collection.name
    
    # Remove from scene hierarchy first
    if collection_name in bpy.context.scene.collection.children:
      bpy.context.scene.collection.children.unlink(collection)
    
    # Remove from blend data
    bpy.data.collections.remove(collection)
    print(f"CLEANUP: Removed empty collection: {collection_name}")
  
  print(f"SUCCESS: Cleaned up {len(collections_to_remove)} empty collections")


def create_spherified_urdf(urdf_path, spheres_dict, output_path):
    """
    Create a modified URDF file where collision geometries are replaced with spheres.
    
    Args:
        urdf_path: Path to original URDF file
        spheres_dict: Dictionary of spheres organized by link name
        output_path: Path for the output URDF file
    """
    print(f"Creating spherified URDF: {output_path}")
    
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_dir = Path(urdf_path).parent
    
    total_spheres = 0
    
    for link in root.findall('link'):
        link_name = link.get('name')
        
        if link_name not in spheres_dict:
            continue
            
        spheres = spheres_dict[link_name]
        if not spheres:
            continue
        
        for collision in link.findall('collision'):
            link.remove(collision)
        
        for i, sphere in enumerate(spheres):
            collision_elem = ET.SubElement(link, 'collision')
            
            geometry_elem = ET.SubElement(collision_elem, 'geometry')
            sphere_elem = ET.SubElement(geometry_elem, 'sphere')
            sphere_elem.set('radius', str(sphere['radius']))
            
            origin_elem = ET.SubElement(collision_elem, 'origin')
            xyz_str = ' '.join(map(str, sphere['origin']))
            origin_elem.set('xyz', xyz_str)
            origin_elem.set('rpy', '0 0 0')
            
            total_spheres += 1
    
    # for mesh in root.findall('.//mesh'):
    #     filename = mesh.get('filename')
    #     if filename and not filename.startswith('file://'):
    #         if filename.startswith('package://'):
    #             # Remove package:// prefix if present
    #             filename = filename[len('package://'):]
            
    #         # Make absolute path with file:// prefix
    #         abs_path = urdf_dir / filename
    #         mesh.set('filename', f'file://{abs_path.absolute()}')
    
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    print(f"SUCCESS: Created spherified URDF with {total_spheres} spheres")
    print(f"  Output: {output_path}")


def create_visualization_urdfs(urdf_path):
    """
    Create visualization URDFs with collision and self-collision spheres.
    Automatically finds sphere JSON files and creates four output URDFs
    two with relative filepaths, two with absolute filepaths, both groups
    including one file with the collision spheres and one with the
    self-collision spheres.
    
    Args:
        urdf_path: Path to original URDF file
    """
    print(f"\n=== CREATING VISUALIZATION URDFS ===")
    
    urdf_path = Path(urdf_path)
    urdf_dir = urdf_path.parent
    robot_name = urdf_path.stem
    
    collision_spheres_dir = urdf_dir / "collision_spheres"
    collision_spheres_path = collision_spheres_dir / "collision_spheres.json"
    self_collision_spheres_path = collision_spheres_dir / "self_collision_spheres.json"
    
    created_files = []
    
    # create collision spheres URDF
    if collision_spheres_path.exists():
        with open(collision_spheres_path, 'r') as f:
            collision_spheres = json.load(f)
        
        output_path = urdf_dir / f"{robot_name}_spheres.urdf"
        create_spherified_urdf(urdf_path, collision_spheres, output_path)
        created_files.append(output_path)
    else:
        print(f"WARNING: No collision_spheres.json found in {collision_spheres_dir}")
    
    # create self-collision spheres URDF
    if self_collision_spheres_path.exists():
        with open(self_collision_spheres_path, 'r') as f:
            self_collision_spheres = json.load(f)
        
        output_path = urdf_dir / f"{robot_name}_selfspheres.urdf"
        create_spherified_urdf(urdf_path, self_collision_spheres, output_path)
        created_files.append(output_path)
    else:
        print(f"WARNING: No self_collision_spheres.json found in {collision_spheres_dir}")
    
    # absolute-path copies
    abs_files = []
    for file in created_files:
      abs_files.append(convert_mesh_paths_to_absolute(file))

    for abs_file in abs_files:
      created_files.append(str(abs_file))  

    if created_files:
        print(f"\nSUCCESS: Created {len(created_files)} visualization URDFs")
        for file in created_files:
            print(f"  {file}")
        print("\nThe URDFs ending in `_abs` can be loaded directly in RViz or Foxglove Studio!")
    else:
        print("ERROR: No sphere files found - cannot create visualization URDFs")
    
    return created_files


def verify_urdf_spheres(urdf_path):
    """
    Verify and analyze a spherified URDF file.
    Shows statistics about spheres per link.
    
    Args:
        urdf_path: Path to spherified URDF file
    """
    print(f"\n=== VERIFYING SPHERIFIED URDF ===")
    print(f"File: {urdf_path}")
    
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        
        total_spheres = 0
        link_stats = []
        
        for link in root.findall('link'):
            link_name = link.get('name')
            collision_count = 0
            sphere_count = 0
            
            for collision in link.findall('collision'):
                collision_count += 1
                geometry = collision.find('geometry')
                if geometry is not None and geometry.find('sphere') is not None:
                    sphere_count += 1
            
            if collision_count > 0:
                link_stats.append([link_name, str(collision_count), str(sphere_count)])
                total_spheres += sphere_count
        
        # Print formatted table
        if link_stats:
            headers = ["Link Name", "Collisions", "Spheres"]
            col_widths = []
            all_rows = [headers] + link_stats
            
            for i in range(len(headers)):
                max_width = max(len(row[i]) for row in all_rows)
                col_widths.append(max_width + 2)
            
            header_format = "".join(f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers)))
            print(f"\n{header_format}")
            print("-" * sum(col_widths))
            
            for row in link_stats:
                row_format = "".join(f"{row[i]:<{col_widths[i]}}" for i in range(len(row)))
                print(row_format)
            
            print(f"\nSUCCESS: URDF contains {total_spheres} sphere collisions across {len(link_stats)} links")
        else:
            print("WARNING: No collision spheres found in URDF")
            
    except Exception as e:
        print(f"ERROR: Failed to parse URDF: {e}")


def batch_create_visualization_urdfs(urdf_directory):
    """
    Create visualization URDFs for all URDF files in a directory.
    
    Args:
        urdf_directory: Directory containing URDF files
    """
    print(f"\n=== BATCH CREATING VISUALIZATION URDFS ===")
    
    urdf_dir = Path(urdf_directory)
    urdf_files = list(urdf_dir.glob("*.urdf"))
    
    if not urdf_files:
        print(f"ERROR: No URDF files found in {urdf_dir}")
        return
    
    print(f"Found {len(urdf_files)} URDF files")
    
    total_created = 0
    for urdf_file in urdf_files:
        print(f"\nProcessing: {urdf_file.name}")
        try:
            created_files = create_visualization_urdfs(urdf_file)
            total_created += len(created_files)
        except Exception as e:
            print(f"ERROR: Failed to process {urdf_file.name}: {e}")
    
    print(f"\nSUCCESS: Created {total_created} visualization URDF files total")


def convert_mesh_paths_to_absolute(urdf_path, output_path=None):
    """
    Convert relative mesh paths in a URDF to absolute file:// paths.
    Useful for making URDFs compatible with RViz and Foxglove.
    
    Args:
        urdf_path: Path to input URDF file
        output_path: Path for output URDF (optional, defaults to input path with _absolute suffix)
    """
    urdf_path = Path(urdf_path)
    
    if output_path is None:
        output_path = urdf_path.parent / f"{urdf_path.stem}_abs.urdf"
    
    print(f"Converting mesh paths to absolute: {urdf_path} -> {output_path}")
    
    # Parse URDF
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_dir = urdf_path.parent
    
    converted_count = 0
    
    # Convert mesh paths
    for mesh in root.findall('.//mesh'):
        filename = mesh.get('filename')
        if filename and not filename.startswith('file://'):
            if filename.startswith('package://'):
                filename = filename[len('package://'):]
            
            abs_path = urdf_dir / filename
            if abs_path.exists():
                mesh.set('filename', f'file://{abs_path.absolute()}')
                converted_count += 1
            else:
                print(f"WARNING: Mesh file not found: {abs_path}")
    
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    print(f"SUCCESS: Converted {converted_count} mesh paths to absolute")
    print(f"  Output: {output_path}")
    
    return output_path


def extract_spheres_from_urdf(urdf_path, output_type='collision'):
    """
    Extract sphere collision geometries from an existing URDF and save to JSON.
    Useful for reverse engineering sphere data from existing spherified URDFs.
    
    Args:
        urdf_path: Path to URDF file containing sphere collision geometries
        output_type: Type of output file ('collision' or 'self_collision')
    """
    print(f"\n=== EXTRACTING SPHERES FROM URDF ===")
    print(f"Input: {urdf_path}")
    print(f"Output type: {output_type}")
    
    urdf_path = Path(urdf_path)
    urdf_dir = urdf_path.parent
    
    # Create output directory
    output_dir = urdf_dir / "collision_spheres"
    output_dir.mkdir(exist_ok=True)
    
    # Parse URDF
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    spheres_dict = {}
    total_spheres = 0
    
    # Extract spheres from each link
    for link in root.findall('link'):
        link_name = link.get('name')
        link_spheres = []
        
        for collision in link.findall('collision'):
            geometry = collision.find('geometry')
            if geometry is not None:
                sphere_elem = geometry.find('sphere')
                if sphere_elem is not None:
                    # Get radius
                    radius = float(sphere_elem.get('radius'))
                    
                    # Get origin
                    origin_elem = collision.find('origin')
                    if origin_elem is not None:
                        xyz_str = origin_elem.get('xyz', '0 0 0')
                        origin = [float(x) for x in xyz_str.split()]
                    else:
                        origin = [0.0, 0.0, 0.0]
                    
                    link_spheres.append({
                        'origin': origin,
                        'radius': radius
                    })
                    total_spheres += 1
        
        if link_spheres:
            spheres_dict[link_name] = link_spheres
    
    # Save to JSON
    if output_type == 'collision':
        output_path = output_dir / "collision_spheres.json"
    else:
        output_path = output_dir / "self_collision_spheres.json"
    
    with open(output_path, 'w') as f:
        json.dump(spheres_dict, f, indent=2)
    
    print(f"SUCCESS: Extracted {total_spheres} spheres from {len(spheres_dict)} links")
    print(f"  Output: {output_path}")
    
    return spheres_dict


####### Legacy functions #######
def add_spheres_from_json(path):
  """Load and add spheres from a JSON file organized per link."""
  with open(path, 'r') as f:
    data = json.load(f)

  if not isinstance(data, dict):
    raise ValueError("Expected a dictionary with link names as keys.")

  counter = 0
  for link_name, link_data in data.items():
    if not isinstance(link_data, dict) or 'spheres' not in link_data:
      continue
    for sphere in link_data['spheres']:
      origin = sphere['origin']
      radius = sphere['radius']
      sphere_obj = add_sphere(*origin, radius)
      sphere_obj.name = f"{link_name}_{counter}"
      counter += 1


def convert_franka_spheres_to_json(spheres):
  """Convert the Franka _SPHERES format into a JSON dict organized per link."""
  out = {}
  for radius, link_dict in spheres:
    for link, arr in link_dict.items():
      if link not in out:
        out[link] = {"spheres": []}
      for coords in arr:
        out[link]["spheres"].append({"origin": list(coords), "radius": radius})
  return out


def convert_self_collision_spheres_to_json(spheres):
  """Convert the _SELF_COLLISION_SPHERES list into a JSON dict organized per link."""
  out = {}
  for link, pos, radius in spheres:
    if link not in out:
      out[link] = {"spheres": []}
    out[link]["spheres"].append({"origin": pos, "radius": radius})
  return out


def debug_sphere_objects(link_name, sphere_type="collision"):
  """
  Debug function to find all sphere objects that match the naming pattern,
  even if they're hidden or in unexpected collections.
  """
  sphere_prefix = f"{link_name}_{sphere_type}_"
  
  print(f"\n=== DEBUGGING SPHERE OBJECTS FOR {link_name} ({sphere_type.upper()}) ===")
  print(f"Searching for objects with prefix: '{sphere_prefix}'")
  
  all_matching_objects = []
  hidden_objects = []
  collection_info = {}
  
  for obj in bpy.data.objects:
    if obj.name.startswith(sphere_prefix):
      all_matching_objects.append(obj)
      
      # Check if object is hidden
      is_hidden = obj.hide_viewport or obj.hide_get()
      if is_hidden:
        hidden_objects.append(obj)
      
      # Check which collections this object belongs to
      obj_collections = [col.name for col in obj.users_collection]
      collection_info[obj.name] = {
        'collections': obj_collections,
        'hidden': is_hidden,
        'location': (obj.location.x, obj.location.y, obj.location.z),
        'radius': obj.dimensions.x / 2 if obj.dimensions.x > 0 else 0.0
      }
  
  print(f"Found {len(all_matching_objects)} objects matching pattern:")
  for obj_name, info in collection_info.items():
    status = "HIDDEN" if info['hidden'] else "VISIBLE"
    collections_str = ", ".join(info['collections']) if info['collections'] else "NO COLLECTIONS"
    loc = info['location']
    radius = info['radius']
    print(f"  {status}: {obj_name}")
    print(f"    Collections: {collections_str}")
    print(f"    Location: ({loc[0]:.3f}, {loc[1]:.3f}, {loc[2]:.3f}), Radius: {radius:.3f}")
  
  if hidden_objects:
    print(f"\nWARNING: Found {len(hidden_objects)} hidden objects that would still be saved!")
    print("These hidden objects might be causing your phantom sphere issue.")
  
  return all_matching_objects, hidden_objects, collection_info


def clean_all_sphere_objects(link_name, sphere_type="collision", confirm=True):
  """
  Remove ALL sphere objects that match the naming pattern, including hidden ones.
  
  Args:
    link_name: The robot link name
    sphere_type: 'collision' or 'self_collision'
    confirm: If True, shows what will be deleted and asks for confirmation
  """
  sphere_prefix = f"{link_name}_{sphere_type}_"
  
  # Find all matching objects
  objects_to_remove = []
  for obj in bpy.data.objects:
    if obj.name.startswith(sphere_prefix):
      objects_to_remove.append(obj)
  
  if not objects_to_remove:
    print(f"SUCCESS: No sphere objects found for {link_name} ({sphere_type})")
    return True
  
  if confirm:
    print(f"\nFOUND {len(objects_to_remove)} SPHERE OBJECTS TO DELETE:")
    for obj in objects_to_remove:
      status = "HIDDEN" if (obj.hide_viewport or obj.hide_get()) else "VISIBLE"
      collections = [col.name for col in obj.users_collection]
      collections_str = ", ".join(collections) if collections else "NO COLLECTIONS"
      print(f"  {status}: {obj.name} (in: {collections_str})")
    
    print(f"\nThis will PERMANENTLY DELETE all {len(objects_to_remove)} objects.")
    response = input("Type 'yes' to confirm deletion: ")
    if response.lower() != 'yes':
      print("CANCELLED: No objects were deleted")
      return False
  
  # Remove all matching objects
  for obj in objects_to_remove:
    print(f"DELETING: {obj.name}")
    bpy.data.objects.remove(obj, do_unlink=True)
  
  print(f"SUCCESS: Deleted {len(objects_to_remove)} sphere objects for {link_name} ({sphere_type})")
  return True


def show_save_preview(link_name, sphere_type="collision"):
  """
  Show what spheres would be saved without actually saving them.
  Helps identify phantom spheres before saving.
  """
  print(f"\n=== SAVE PREVIEW FOR {link_name} ({sphere_type.upper()}) ===")
  
  sphere_prefix = f"{link_name}_{sphere_type}_"
  spheres_to_save = []
  
  for obj in bpy.data.objects:
    if obj.type == 'MESH' and obj.name.startswith(sphere_prefix):
      loc = obj.location
      radius = obj.dimensions.x / 2
      is_hidden = obj.hide_viewport or obj.hide_get()
      collections = [col.name for col in obj.users_collection]
      
      sphere_data = {
        "name": obj.name,
        "origin": [round(loc.x, 6), round(loc.y, 6), round(loc.z, 6)],
        "radius": round(radius, 6),
        "hidden": is_hidden,
        "collections": collections
      }
      spheres_to_save.append(sphere_data)
  
  if not spheres_to_save:
    print("No spheres found to save.")
    return []
  
  print(f"Would save {len(spheres_to_save)} spheres:")
  for i, sphere in enumerate(spheres_to_save):
    status = "HIDDEN" if sphere['hidden'] else "VISIBLE"
    collections_str = ", ".join(sphere['collections']) if sphere['collections'] else "NO COLLECTIONS"
    origin = sphere['origin']
    radius = sphere['radius']
    print(f"  {i}: {status} {sphere['name']}")
    print(f"      Location: ({origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}), Radius: {radius:.3f}")
    print(f"      Collections: {collections_str}")
  
  return spheres_to_save


def unhide_all_spheres(link_name, sphere_type="collision"):
  """
  Make all sphere objects visible (unhide them) to help locate phantom spheres.
  """
  sphere_prefix = f"{link_name}_{sphere_type}_"
  unhidden_count = 0
  
  for obj in bpy.data.objects:
    if obj.name.startswith(sphere_prefix):
      if obj.hide_viewport or obj.hide_get():
        obj.hide_viewport = False
        obj.hide_set(False)
        unhidden_count += 1
        print(f"UNHIDDEN: {obj.name}")
  
  if unhidden_count > 0:
    print(f"SUCCESS: Made {unhidden_count} hidden sphere objects visible")
  else:
    print("No hidden sphere objects found")
  
  return unhidden_count


def clean_orphaned_spheres(link_name, sphere_type="collision"):
  """
  Remove sphere objects that match the naming pattern but aren't in the proper collection.
  This fixes the issue where spheres exist outside collections and cause duplicates.
  """
  sphere_prefix = f"{link_name}_{sphere_type}_"
  expected_collection_name = f"{link_name}_{sphere_type}_spheres"
  
  orphaned_objects = []
  
  for obj in bpy.data.objects:
    if obj.name.startswith(sphere_prefix):
      # Check if this object is in the expected collection
      obj_collections = [col.name for col in obj.users_collection]
      
      if expected_collection_name not in obj_collections:
        # This sphere is orphaned (not in the proper collection)
        orphaned_objects.append(obj)
  
  if not orphaned_objects:
    print(f"SUCCESS: No orphaned spheres found for {link_name} ({sphere_type})")
    return True
  
  print(f"FOUND {len(orphaned_objects)} ORPHANED SPHERE OBJECTS:")
  for obj in orphaned_objects:
    collections_str = ", ".join([col.name for col in obj.users_collection]) if obj.users_collection else "NO COLLECTIONS"
    print(f"  {obj.name} (in: {collections_str})")
  
  # Remove orphaned objects
  for obj in orphaned_objects:
    print(f"REMOVING ORPHANED: {obj.name}")
    bpy.data.objects.remove(obj, do_unlink=True)
  
  print(f"SUCCESS: Cleaned up {len(orphaned_objects)} orphaned sphere objects")
  return True


def move_object_to_collection_safe(obj, target_collection):
  """
  Safely move an object to a collection with proper error handling and cleanup.
  """
  try:
    # Remove from ALL collections first (including Scene Collection)
    for collection in obj.users_collection[:]:  # Copy list to avoid modification during iteration
      collection.objects.unlink(obj)
    
    # Add to target collection
    target_collection.objects.link(obj)
    
    # Verify the move was successful
    if target_collection not in obj.users_collection:
      print(f"WARNING: Failed to move {obj.name} to {target_collection.name}")
      return False
    
    return True
    
  except Exception as e:
    print(f"ERROR: Failed to move {obj.name} to collection: {e}")
    return False
