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

echo -e "${GREEN}Starting RViz...${NC}"
echo -e "${YELLOW}Using URDF: $URDF_FILE${NC}"

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

# Wait a bit more
sleep 2

# Start RViz
echo -e "${GREEN}Starting RViz...${NC}"
echo -e "${YELLOW}In RViz:${NC}"
echo -e "  1. Add 'RobotModel' display"
echo -e "  2. Set Fixed Frame to the base link"
echo ""
rviz2 &
RVIZ_PID=$!

echo -e "${YELLOW}Use the Joint State Publisher GUI to move the robot${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop all services.${NC}"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Stopping all processes...${NC}"
    pkill -f "rviz2"
    pkill -f "ros2 run robot_state_publisher"
    pkill -f "ros2 run joint_state_publisher_gui"
    exit 0
}

# Setup signal handlers
trap cleanup SIGINT SIGTERM

# Wait for all background processes
wait $STATE_PUB_PID $JOINT_GUI_PID $RVIZ_PID 