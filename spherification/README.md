# Spherification Utils

A comprehensive toolkit for creating and editing collision spheres for robotic systems using Blender as the visualization and editing environment.

## Core Functions

### Robot Analysis
- `analyze_robot_spheres(urdf_path)` - Show formatted table of links and sphere counts

### Sphere Generation
- `generate_initial_spheres(urdf_path)` - Generate spheres using [foam](https://github.com/CoMMALab/foam/tree/master) for new robots
- `parse_foam_output(foam_output_path)` - Parse foam's JSON output format

### Interactive Editing
- `load_link_for_editing(urdf_path, link_name, sphere_type='collision')` - Load mesh and spheres into Blender
- `save_edited_spheres(urdf_path, link_name, sphere_type='collision')` - Save edited spheres back to JSON

### URDF Generation
- `create_visualization_urdfs(urdf_path)` - Create collision and self-collision visualization URDFs for use with RViz or Foxglove Studio
- `create_spherified_urdf(urdf_path, spheres_dict, output_path)` - Create URDF with sphere collision geometries
- `verify_urdf_spheres(urdf_path)` - Verify and analyze spherified URDF files
- `convert_mesh_paths_to_absolute(urdf_path)` - Convert mesh paths to absolute file:// paths (often required by RViz and Foxglove Studio)

## Standard JSON format for collision spheres

```json
{
    "first_link_name": [
        {
            "origin": [
                <x>,
                <y>,
                <z>
            ],
            "radius": <radius>
        },
        [more spheres ...]
    ],
    "second_link_name": [
        {
            "origin": [
                <x>,
                <y>,
                <z>
            ],
            "radius": <radius>
        },
        [more spheres ...]
    ]
}
```

All units in meters. These `.json` files are placed in `collision_spheres/` in the directory of the robot's urdf,
from which the link names are taken.

## Sphere Types

- `"collision"` - Regular collision detection spheres
- `"self_collision"` - Self-collision detection spheres (typically larger, fewer)

It can be important to have both these sphere representations, as the needs for collision detection with the environment, and self-collision between the robots own non-adjacent links can differ.
The regular `collision` spheres may not need to cover every vertex of the visual link mesh, it's up to your specification. Also, you may not want to cover the end-effector's contact surfaces. For the `self_collision` spheres, you will want to cover the full end-effector, and may want to be a bit more conservative, not letting any vertices peek through. 

## File Structure

The file structure after using the spherification utilities will look like this:

```
robot_directory/
├── robot.urdf                           # Original URDF
├── robot_collision_spheres.urdf         # Visualization URDF (collision)
├── robot_self_collision_spheres.urdf    # Visualization URDF (self-collision)
├── collision_spheres/
│   ├── collision_spheres.json           # Collision spheres
│   └── self_collision_spheres.json      # Self-collision spheres
└── meshes/
    ├── visual/                          # Visual meshes
    └── collision/                       # Collision meshes
```

# Blender Sphere Editing Workflow

This guide explains how to use the sphere editing tools in Blender for robot collision sphere generation and refinement.

## Prerequisites

1. Blender 4.2 LTS installed
2. Robot URDF file with mesh references, and the references meshes
3. Optional: collision_spheres.json and self_collision_spheres.json files

## Quick Start

### 1. Open Blender and Import Module

In Blender's Python console (click Scripting tab at the top):

```python
import sys
sys.path.append('/<absolute_path_to>/spherification')
import spherification_utils
```

### 2. Analyze Your Robot

```python
# Analyze robot structure and existing spheres
spherification_utils.analyze_robot_spheres('path/to/your/robot.urdf')
```

This will show a properly formatted table like:
```
Link Name          Visual Mesh  Collision  Self-Collision  
-----------------------------------------------------------
link_name0         Yes          1          2               
link_name1         Yes          4          4               
...
```

The last two columns show the number of spheres of each type.

### 3. Generate Initial Spheres (if needed)

If you don't have sphere files yet:

```python
# Generate spheres using foam for all meshes
spherification_utils.generate_initial_spheres('path/to/your/robot.urdf')
```

This will take a few minutes on a modern desktop PC. The Blender interactive shell might appear to freeze up until it is finished.

### 4. Load Link for Editing

Set "Material Preview" as the Viewport Shading mode (top right of blender viewport) to see the collision spheres in green.

Load a specific link with its mesh and spheres:

```python
# For collision spheres
spherification_utils.load_link_for_editing('path/to/your/robot.urdf', 'link_name', 'collision')

# For self-collision spheres  
spherification_utils.load_link_for_editing('path/to/your/robot.urdf', 'link_name', 'self_collision')

# Disable coordinate fix if needed (adjusts Y-up to Z-up for .obj files)
spherification_utils.load_link_for_editing('path/to/your/robot.urdf', 'link_name', 'collision', apply_coordinate_fix=False)
```

This will:
- Clear the scene
- Load the link's mesh (locked in place)
- Add all spheres for that link (green, editable)

Adjust the view by dragging with middle-click and shift+middle-click. Dragging with alt+middle-click will snap to the coordinate planes, which is helpful for precise editing.


### 5. Edit Spheres in Blender

Use standard Blender controls:
- **Select**: Left-click on spheres
- **Move**: Press `G`, then optionally `X`/`Y`/`Z` for axis-constrained movement
- **Scale**: Press `S` to scale sphere size
- **Delete**: Press `X` > Delete to remove spheres
- **Add**: Use the `add_sphere(x,y,z,radius)` function or duplicate existing ones (make sure they follow the same naming scheme, `{link_name}_{sphere_type}_{index}`, where `sphere_type` is either `collision` or `self_collision`)

Tip: Try your best to have as few unique sphere radii as possible. In Avoid Everything, the spheres are grouped by radius, and the fewer unique radii there are, the faster can batched signed distance field computations on the spheres be completed.
E.g. instead of having one sphere have radius 0.046 and another 0.047, make them both have either 0.046 or 0.047 -> fewer unique radii.

Radii are currently rounded to 3 decimals in this fork of Avoid Everything, so storing any further decimals is superfluous.

### 6. Save Your Changes

```python
# Save collision spheres
spherification_utils.save_edited_spheres('path/to/your/robot.urdf', 'link_name', 'collision')

# Save self-collision spheres
spherification_utils.save_edited_spheres('path/to/your/robot.urdf', 'link_name', 'self_collision')
```

I'd suggest editing all `collision` links and when you're satisfied, copy the contents of collision_spheres.json into self_collision_spheres.json, and then edit the `self_collision` links to just be a bit more conservative.
After loading and saving each link that you want to have collision spheres, 
the collision_spheres.json and self_collision_spheres.json files should be ready for use.

### 7. (Optional) Create urdfs with collision spheres for visualization

Make sure you have the collision_spheres.json and self_collision_spheres.json files ready at this step.

```
create_visualization_urdfs('path/to/robot.urdf')
```

This will create:
- `robot_spheres.urdf` - URDF with collision spheres
- `robot_selfspheres.urdf` - URDF with self-collision spheres
- `robot_spheres_abs.urdf` - URDF with collision spheres and absolute filepaths
- `robot_selfspheres_abs.urdf` - URDF with self-collision spheres and absolute filepaths

The URDFs with absolute file paths can be loaded directly in **RViz** and **Foxglove Studio** for visualization! For visualization with **viz_server**, the regular URDFs with relative paths should be used.

To visualize in RViz or Foxglove, you can do the following:

In one terminal, set the urdf contents as an environment variable and run the robot state publisher:
```
export SPHERIZED_ROBOT_DESCRIPTION="$(cat assets/robot/robot_spheres_abs.urdf)"
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:="$SPHERIZED_ROBOT_DESCRIPTION"
```

Open another terminal and run
```
ros2 run joint_state_publisher_gui joint_state_publisher_gui
```

Then in a third terminal run RViz, or Foxglove Studio (native application).
```
ros2 run rviz2 rviz2  # RViz for example
```

In both applications, you should be able to toggle viewing the collision meshes, and see the spheres.  


## Tips

1. **Use collision meshes as basis for foam sphere generation** - If your urdf has simplified collision meshes already, generating spheres from them will usually give you better results than using the detailed visual meshes. 
2. **Use sphere hierarchy** - Large spheres for coarse coverage, small ones for details.
3. **Self-collision spheres** - Sometimes need to be larger, more conservative. See the Franka robot for reference.