# Autonomous Exploration Mission for Boston Dynamics Spot

## Overview

This project implements an autonomous exploration system for the Boston Dynamics Spot robot. The system enables the robot to autonomously explore and map unknown environments using a grid-based approach combined with GraphNav navigation, obstacle avoidance, and intelligent pathfinding.

## Key Features

### Autonomous Exploration
- **Grid-based environment mapping**: Divides the environment into a configurable grid of cells
- **Serpentine path planning**: Generates optimal lawnmower-pattern exploration paths
- **Intelligent frontier detection**: Identifies unexplored regions using frontier-based exploration
- **Adaptive navigation**: Switches between direct movement and waypoint-based navigation based on accessibility

### Obstacle Avoidance
- **Real-time obstacle detection**: Uses Spot's local grid sensors to detect no-step zones
- **Line-of-sight verification**: Samples multiple points within target cells to find clear paths
- **Dynamic path adjustment**: Marks blocked cells and adapts exploration strategy

### GraphNav Integration
- **Automatic waypoint creation**: Creates manual waypoints at explored cell positions
- **Edge-based graph topology**: Maintains accurate connectivity between waypoints
- **Shortest path navigation**: Computes optimal paths through the waypoint graph
- **Graph persistence**: Saves exploration graphs for future missions

### Visualization System
- **Real-time monitoring**: Displays exploration progress with overlaid maps
- **Mission recording**: Saves visualization snapshots to timestamped folders
- **Graph topology visualization**: Shows waypoint connections based on actual graph edges
- **Path tracking**: Distinguishes between exploration and navigation movements

## Architecture

### Core Components

#### 1. `easy_walk.py`
Main execution script containing:
- Mission initialization and robot control
- Frontier-based exploration algorithm
- Cell sampling and path validation
- Integration of all subsystems

#### 2. `environmentMap.py`
Environment representation module:
- Grid-based map management (visited/blocked/unknown cells)
- World-to-grid coordinate transformations
- Cell side exploration tracking (NESW encoding)
- Serpentine path generation
- Waypoint and robot path recording

#### 3. `navGraphUtils.py`
GraphNav interface and utilities:
- Waypoint creation with cell metadata
- Graph recording and downloading
- Navigation to waypoints with localization recovery
- Shortest path computation
- Edge creation and graph topology management
- Fiducial-based initialization

#### 4. `spotUtils.py`
Robot state utilities:
- Position extraction from robot state
- Coordinate frame transformations

#### 5. `movements.py`
Robot movement primitives:
- Relative movements in different frames
- Rotation commands
- Movement validation

#### 6. `spotGrid.py`
Local grid processing:
- No-step grid extraction from Spot sensors
- Point cloud visualization
- Obstacle detection

#### 7. `spotLogInUtils.py`
Robot connection utilities:
- Authentication and lease management
- E-stop handling
- Client initialization

## Installation

### Recommended: Create a Conda Environment

It is strongly recommended to create a dedicated Conda environment for this project to avoid dependency conflicts:

```bash
# Create a new Conda environment
conda create -n spot-exploration python=3.8

# Activate the environment
conda activate spot-exploration
```

All subsequent installation steps should be performed within this activated environment.

### Prerequisites

- Python 3.7 or higher
- Boston Dynamics Spot SDK installed
- Spot robot with network connectivity
- Valid Spot credentials
- Anaconda or Miniconda (recommended)

### Dependencies

Install the required Boston Dynamics Spot SDK:

```bash
pip install bosdyn-client bosdyn-core bosdyn-api
```

Additional required packages:
- `numpy`: Numerical computations and grid operations
- `matplotlib`: Real-time visualization and map rendering

Install additional packages:

```bash
pip install numpy matplotlib
```

## Usage

### Basic Execution

```python
python easy_walk.py --username USER --password PASS ROBOT_IP
```

### Configuration Parameters

The system can be configured by modifying the `EnvironmentMap` initialization in `easy_walk.py`:

```python
env = environmentMap.EnvironmentMap(
    rows=4,        # Number of rows in exploration grid
    cols=4,        # Number of columns in exploration grid
    cell_size=1.0  # Size of each cell in meters
)
```

### Mission Workflow

1. **Initialization**
   - Robot authentication and lease acquisition
   - Recording service initialization
   - Optional fiducial-based localization
   - First waypoint creation at origin

2. **Exploration Loop**
   - Identify adjacent frontier cells
   - Select target based on path rank
   - Sample candidate points in target cell
   - Validate clear line-of-sight
   - Execute movement or navigate via waypoint
   - Create waypoint upon successful entry
   - Update map and frontier list

3. **Completion**
   - Navigate to first waypoint
   - Sit and release lease
   - Download complete graph with snapshots

## Output Structure

```
autonomousMission/
├── MissionMap/
│   └── Mission-YYYYMMDD_HHMMSS/
│       ├── iteration_0_sampling.png
│       ├── iteration_1_sampling.png
│       └── ...
├── downloaded_graph/
│   ├── graph
│   ├── waypoint_snapshots/
│   └── edge_snapshots/
└── ...
```

### Mission Maps
Each exploration mission creates a timestamped folder containing:
- Visualization snapshots at each cell sampling iteration
- Overlaid global and local grids
- Waypoint connections (edge-based topology)
- Robot path traces (exploration vs navigation)

### Graph Files
Downloaded graphs include:
- Graph structure (waypoints and edges)
- Waypoint snapshots (sensor data, images)
- Edge snapshots (transformation data)

## Algorithm Details

### Frontier-Based Exploration

The system uses a rank-based frontier approach:

1. **Frontier Detection**: Identifies unvisited cells adjacent to visited cells
2. **Rank Assignment**: Assigns rank based on position in serpentine path
3. **Target Selection**: 
   - Adjacent frontiers: lowest rank
   - Distant frontiers: navigate via nearest waypoint
4. **Adaptive Strategy**: Switches between direct movement and waypoint navigation

### Cell Sampling Strategy

For each target cell:
1. Generate 20 random sample points within cell bounds
2. Check line-of-sight from robot to each sample
3. Select point closest to cell center with clear path
4. If no clear path found, mark cell as blocked

### Distance Calculations

**Important**: The system uses consistent coordinate frames for distance calculations:
- Robot positions: VISION frame (from `spotUtils.getPosition()`)
- Waypoint positions: Saved in VISION frame during creation
- Nearest waypoint search: Euclidean distance in VISION frame
- Cell-based search: Manhattan distance in grid coordinates

### Graph Topology

Waypoints are connected by edges that represent:
- Sequential exploration movements
- Navigation paths between distant cells
- Manually created connections for optimization

Visualization shows only actual graph edges, not sequential ordering.

## Troubleshooting

### Common Issues

**Robot loses localization during navigation**
- System automatically attempts re-localization
- Falls back to previous waypoint if recovery fails
- Consider reducing cell size for denser waypoint placement

**No clear path to target cell**
- Cell is marked as blocked (-1)
- Frontier is updated to exclude blocked cell
- Exploration continues with next frontier

**Waypoint navigation fails**
- Check GraphNav service status
- Verify graph download completed successfully
- Ensure waypoint IDs are valid

## Technical Notes

### Coordinate Frames

The system uses multiple coordinate frames:
- **VISION**: Primary frame for world coordinates
- **ODOM**: Odometry frame (kinematic odometry)
- **BODY**: Robot body frame
- Grid coordinates are relative to origin with yaw rotation

### Cell Side Encoding

Each cell tracks which sides have been explored using 4-bit encoding:
- Bit 3 (0b1000): North
- Bit 2 (0b0100): East
- Bit 1 (0b0010): South
- Bit 0 (0b0001): West

### Waypoint Naming Convention

- Manual waypoints: `wp_0`, `wp_1`, `wp_2`, ...
- Automatically numbered sequentially
- Cell metadata saved for distance calculations

## Performance Considerations

- **Grid size**: Larger cells reduce waypoint density but may miss narrow passages
- **Sample count**: More samples improve path finding but increase computation time
- **Visualization frequency**: Real-time display impacts execution speed (0.5s pause per iteration)
- **Graph downloads**: Full downloads with snapshots can be slow on large graphs

## Development

### Adding Custom Behaviors

To extend the exploration strategy:

1. Modify frontier selection in `easy_walk.py`:
```python
# Custom frontier ranking logic
def custom_frontier_rank(cell, env):
    return your_ranking_function(cell)
```

2. Add new movement primitives in `movements.py`
3. Extend map representation in `environmentMap.py`

### Debugging

Enable verbose logging in key functions:
- `find_nearest_waypoint_to_position()`: Distance calculations
- `attempt_enter_cell_from_position()`: Cell sampling details
- `navigate_shortest_path()`: Pathfinding steps

## License

This project is developed for research purposes as part of the Spot SDK ecosystem.

## References

- [Boston Dynamics Spot SDK Documentation](https://dev.bostondynamics.com/)
- [GraphNav Service Documentation](https://dev.bostondynamics.com/docs/concepts/autonomy/graphnav_service)
- Frontier-based exploration algorithms in mobile robotics

## Contributors

Developed as part of autonomous robotics research at Università degli Studi di Firenze.

## Version History

- Current version: Integration of grid-based exploration with GraphNav
- Recent improvements:
  - Fixed waypoint distance calculations (VISION frame consistency)
  - Edge-based waypoint visualization
  - Mission-specific output folders
  - Real-time monitoring with persistent saving

