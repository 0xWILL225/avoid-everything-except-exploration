#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if URDF file argument is provided
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No URDF file provided${NC}"
    echo -e "${YELLOW}Usage: $0 <path_to_urdf_file>${NC}"
    exit 1
fi

URDF_FILE="$1"

# Source ROS2 environment
source /opt/ros/humble/setup.bash

# Check if URDF file exists
if [ ! -f "$URDF_FILE" ]; then
    echo -e "${RED}Error: URDF file not found at $URDF_FILE${NC}"
    exit 1
fi

echo -e "${YELLOW}Using URDF: $URDF_FILE${NC}"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Stopping all processes...${NC}"
    pkill -f "foxglove-studio"
    pkill -f "ros2 run foxglove_bridge"
    pkill -f "ros2 run robot_state_publisher"
    pkill -f "ros2 run joint_state_publisher_gui"
    exit 0
}

# Setup signal handlers
trap cleanup SIGINT SIGTERM

# Start Foxglove Studio in background
echo -e "${GREEN}Starting Foxglove Studio...${NC}"
foxglove-studio &
FOXGLOVE_PID=$!

# Wait a bit for Foxglove to start
sleep 3

# Start the Foxglove bridge (WebSocket server)
echo -e "${GREEN}Starting Foxglove Bridge (WebSocket server)...${NC}"
ros2 run foxglove_bridge foxglove_bridge --ros-args -p port:=8765 &
BRIDGE_PID=$!

# Wait a bit for the bridge to start
sleep 2

# Load and publish robot description
echo -e "${GREEN}Loading robot description...${NC}"
export ROBOT_DESCRIPTION="$(cat $URDF_FILE)"
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:="$ROBOT_DESCRIPTION" &
STATE_PUB_PID=$!

# Wait a bit for robot state publisher to start
sleep 2

# Start joint state publisher GUI
echo -e "${GREEN}Starting Joint State Publisher GUI...${NC}"
ros2 run joint_state_publisher_gui joint_state_publisher_gui &
JOINT_GUI_PID=$!

echo -e "${GREEN}=== All services started successfully! ===${NC}"
echo -e "${YELLOW}Foxglove Studio should now be running with the websocket bridge.${NC}"
echo -e "${YELLOW}To connect Foxglove to ROS2:${NC}"
echo -e "  First, log in if prompted (only required once)"
echo -e "  1. In Foxglove Studio, click 'Open Connection'"
echo -e "  2. Select 'Foxglove WebSocket'"
echo -e "  3. Use URL: ws://localhost:8765"
echo -e "  4. Click 'Open'"
echo ""
echo -e "${YELLOW}The Joint State Publisher GUI should be open to control the robot.${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop all services.${NC}"
echo ""

# Wait for all background processes
wait $FOXGLOVE_PID $BRIDGE_PID $STATE_PUB_PID $JOINT_GUI_PID 