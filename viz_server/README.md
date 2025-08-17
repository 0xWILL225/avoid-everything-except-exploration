# viz_server

## Overview
The viz_server package is a ROS2 visualization server that bridges ZeroMQ with ROS2 topics for use with RViz or Foxglove Studio. It is tailored to the visualization needed in the Avoid Everything project.

## Features

- **Robot Visualization**: Publish robot joint states and trajectories
- **Point Cloud Visualization**: Separate topics for robot, target, and obstacle point clouds
- **Ghost Visualizations**: Display translucent robot meshes at specific poses or configurations
- **Obstacle Visualizations**: Display cuboid and cylinder obstacles
- **Universal World Frame**: Automatic static transform from "world" to robot base link for consistent RViz setup

## Usage

Run Rviz2, preferrably before starting the server.

```
ros2 run rviz2 rviz2
```

### Server

The server is started automatically when you call connect() from any client.
You can turn it off by calling shutdown() on the client, or by running the `shutdown_viz_server.py` script.
The server will run a `robot_state_publisher` process with the same lifetime as the server, when the server is shutdown, so is the `robot_state_publisher`.

### Client API
```python
import viz_client

urdf_path = "/path/to/urdf/robot.urdf"
viz_client.connect(urdf_path)

config = {
    "joint1": 0.0,
    "joint2": -0.785,
    "joint3": 0.0,
    "joint4": -2.356,
    "joint5": 0.0,
    "joint6": 1.571,
    "joint7": 0.785,
    "finger_joint1": 0.02, 
    # "finger_joint2": 0.02  # if "finger_joint2" mimics "finger_joint1", only need to send one of them
}

# Publish joint states
viz_client.publish_joints(config)


# Publish trajectory of joint waypoints
config2 = config.copy()
config2["joint2"] = -0.185
waypoints = [config, config2]
viz_client.publish_trajectory(waypoints)

# Publish point clouds for the robot, target and obstacles
robot_points = np.random.rand(200, 3).astype(np.float32)
target_points = np.random.rand(150, 3).astype(np.float32) + [1, 0, 0]
obstacle_points = np.random.rand(100, 3).astype(np.float32) + [-1, 0, 0]

viz_client.publish_robot_pointcloud(robot_points)
viz_client.publish_target_pointcloud(target_points)
viz_client.publish_obstacle_pointcloud(obstacle_points)

# Ghost visualizations
viz_client.publish_ghost_end_effector(pose, color=[0, 1, 0], alpha=0.5)
viz_client.publish_ghost_robot(config, color=[1, 0, 0], alpha=0.3)

# Clear visualizations
viz_client.clear_robot_pointcloud()
viz_client.clear_target_pointcloud()
viz_client.clear_obstacle_pointcloud()
viz_client.clear_ghost_end_effector()
viz_client.clear_ghost_robot()
```

## ROS2 Topics

- `/joint_states` - Robot joint states
- `/viz/robot_points` - Robot point cloud
- `/viz/target_points` - Target point cloud  
- `/viz/obstacle_points` - Obstacle point cloud
- `/viz/markers` - Ghost visualization markers
- `/tf_static` - Static transform (world → base_link)

## RViz Configuration

The server publishes a static transform from "world" to the robot's base link, allowing you to set "world" as the fixed frame in RViz for universal compatibility across different robots. The default RViz configuration in the Docker devcontainer is set up for the viz_server publishers.

## Dependencies

- ROS2 Humble
- ZeroMQ (pyzmq)
- NumPy
- SciPy
- urdf_parser_py
- termcolor
