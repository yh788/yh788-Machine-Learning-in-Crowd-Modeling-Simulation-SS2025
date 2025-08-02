import json
import os

import numpy as np
import csv

CONFIG_FOLDER = "configs"
OUTPUT_FOLDER = "outputs"

CELL_SIZE = 0.4

# need to create folders if they don't exist
os.makedirs(CONFIG_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def get_vertical_object(x: int, y_start: int, y_end: int) -> tuple[dict]:
    """Generates a list of positions in the provided vertical range.

    Parameters:
    -----------
    x : int
        The x coordinate of the vertical object.
    y_start : int
        The y coordinate of the lower end of the object.
    y_end : int
        The y coordinate of the upped end of the object.

    Returns:
    --------
    tuple[dict]
        A list of dictionaries with keys ('x', 'y') specifying positions
        of the object entries. Both ends at y_start and y_end are
        included to the object.
    """
    return tuple({"x": x, "y": y} for y in range(y_start, y_end + 1))


def get_horizontal_object(x_start: int, x_end: int, y: int) -> tuple[dict]:
    """Generates a list of positions in the provided horizontal range.

    Parameters:
    -----------
    x_start : int
        The x coordinate of the left end of the object.
    x_end : int
        The x coordinate of the right end of the object.
    y : int
        The y coordinate of the horizontal object.

    Returns:
    --------
    tuple[dict]
        A list of dictionaries with keys ('x', 'y') specifying positions
        of the object entries. Both ends at x_start and x_end are
        included to the object.
    """
    return tuple({"x": x, "y": y} for x in range(x_start, x_end + 1))


def generate_pedestrians(
    hor_span: tuple[int, int],
    vert_span: tuple[int, int],
    n_pedestrians: int,
    speed_bounds: tuple[float, float] = (1, 1),
    random_seed: int = 42,
) -> list[dict]:
    """Generates pedestrians in the specified rectangular area.

    Parameters:
    -----------
    hor_span : Tuple[int, int]
        A tuple (x_min, x_max) specifying horizontal borders of the
        spawn area.
    vert_span : Tuple[int, int]
        A tuple (y_min, y_max) specifying vertical borders of the
        spawn area.
    n_pedestrians : int
        The number of pedestrians to generate. The positions of the
        pedestrians are sampled from the uniform distribution.
    speed_bounds : Tuple[float, float] = (1, 1)
        A tuple (speed_min, speed_max) specifying speed bounds of
        pedestrians. The unit of speed is cells/step. A speed value for
        each pedestrian is sampled from the uniform distribution.
    random_seed : int = 42
        The random seed used to define a random generator.

    Returns:
    --------
    list[dict]
        A list of dictionaries with keys ('ID', 'x', 'y', 'speed')
        specifying the initial configuration of pedestrians in the
        simulation.
    """
    rng = np.random.default_rng(random_seed)
    spawn_width = hor_span[1] - hor_span[0] + 1
    spawn_height = vert_span[1] - vert_span[0] + 1

    positions = rng.choice(
        spawn_width * spawn_height, size=n_pedestrians, replace=False
    )
    xs = positions % spawn_width + hor_span[0]
    ys = positions // spawn_width + vert_span[0]
    speeds = rng.uniform(*speed_bounds, size=n_pedestrians)
    pedestrians = []
    for i, (x, y, speed) in enumerate(zip(xs, ys, speeds)):
        pedestrians.append({"ID": i, "x": int(x), "y": int(y), "speed": speed})
    return pedestrians


def save_json(
    filename: str,
    grid_size: dict,
    targets: tuple[dict],
    measuring_points: tuple[dict],
    obstacles: tuple[dict],
    pedestrians: tuple[dict],
    is_absorbing: bool,
    distance_computation: str,
    output_filename: str,
):
    """Saves the simulation configuration to a .json file."""

    config = {
        "grid_size": grid_size,
        "targets": targets,
        "obstacles": obstacles,
        "pedestrians": pedestrians,
        "measuring_points": measuring_points,
        "is_absorbing": is_absorbing,
        "distance_computation": distance_computation,
        "output_filename": output_filename,
    }
    with open(filename, "w") as fout:
        json.dump(config, fout, sort_keys=True, indent=4)


def task_1(filename: str):
    """Saves a configuration file for the Task 1."""

    config_filename = os.path.join(CONFIG_FOLDER, f"{filename}.json")
    grid_size = {"width": 5, "height": 5}
    targets = [{"x": 3, "y": 2}]
    measuring_points = []
    obstacles = []
    pedestrians = [{"ID": 1, "x": 1, "y": 1, "speed": 1}]
    is_absorbing = False
    distance_computation = "naive"
    output_filename = os.path.join(OUTPUT_FOLDER, f"{filename}.csv")

    save_json(
        config_filename,
        grid_size,
        targets,
        measuring_points,
        obstacles,
        pedestrians,
        is_absorbing,
        distance_computation,
        output_filename,
    )


def task_2(filename: str):
    """Saves a configuration file for the Task 2."""

    config_filename = os.path.join(CONFIG_FOLDER, f"{filename}.json")
    grid_size = {"width": 50, "height": 50}
    targets = [{"x": 25, "y": 25}]
    measuring_points = []
    obstacles = []
    # want a pedestrian at (5,25)
    pedestrians = [{"ID": 1, "x": 5, "y": 25, "speed": 1}]
    is_absorbing = False
    distance_computation = "naive"
    output_filename = os.path.join(OUTPUT_FOLDER, f"{filename}.csv")

    save_json(
        config_filename,
        grid_size,
        targets,
        measuring_points,
        obstacles,
        pedestrians,
        is_absorbing,
        distance_computation,
        output_filename,
    )


def task_4_simple(filename: str):
    """Configuration for the simple obstacle test (Task 4)."""

    config_filename = os.path.join(CONFIG_FOLDER, f"{filename}.json")
    grid_size = {"width": 50, "height": 50}
    targets = [{"x": 25, "y": 25}]
    measuring_points = []
    obstacles = [
        {"x": 22, "y": 25},
        {"x": 23, "y": 25},
        {"x": 22, "y": 24},
        {"x": 23, "y": 24},
    ]
    pedestrians = [{"ID": 1, "x": 5, "y": 25, "speed": 1}]
    is_absorbing = False
    distance_computation = "dijkstra"
    output_filename = os.path.join(OUTPUT_FOLDER, f"{filename}.csv")

    save_json(
        config_filename,
        grid_size,
        targets,
        measuring_points,
        obstacles,
        pedestrians,
        is_absorbing,
        distance_computation,
        output_filename,
    )


def task_4_chicken(filename: str):
    """Configuration for the chicken test (Task 4)."""

    config_filename = os.path.join(CONFIG_FOLDER, f"{filename}.json")
    grid_size = {"width": 50, "height": 50}
    targets = [{"x": 25, "y": 25}]
    measuring_points = []
    # Obstacle is in reverse C-shape.
    obstacles = [
        {"x": 14, "y": 21},
        {"x": 15, "y": 21},
        {"x": 16, "y": 21},
        {"x": 17, "y": 21},
        {"x": 18, "y": 21},
        {"x": 19, "y": 21},
        {"x": 19, "y": 22},
        {"x": 19, "y": 23},
        {"x": 19, "y": 24},
        {"x": 19, "y": 25},
        {"x": 19, "y": 26},
        {"x": 19, "y": 27},
        {"x": 19, "y": 28},
        {"x": 19, "y": 29},
        {"x": 14, "y": 29},
        {"x": 15, "y": 29},
        {"x": 16, "y": 29},
        {"x": 17, "y": 29},
        {"x": 18, "y": 29},
    ]
    pedestrians = [{"ID": 1, "x": 5, "y": 25, "speed": 1}]
    is_absorbing = False
    distance_computation = "dijkstra"
    output_filename = os.path.join(OUTPUT_FOLDER, f"{filename}.csv")

    save_json(
        config_filename,
        grid_size,
        targets,
        measuring_points,
        obstacles,
        pedestrians,
        is_absorbing,
        distance_computation,
        output_filename,
    )


def task_4_bottleneck(filename: str):
    """Bottleneck test with 50 non-overlapping pedestrians."""
    config_filename = os.path.join(CONFIG_FOLDER, f"{filename}.json")
    grid_size = {"width": 50, "height": 30}
    targets = [{"x": 49, "y": 13}, {"x": 49, "y": 14}, {"x": 49, "y": 15}]
    measuring_points = []

    obstacles = []
    # Create outside walls
    for x in range(50):
        if x not in range(21, 30):
            obstacles.append({"x": x, "y": 29})
            obstacles.append({"x": x, "y": 0})

    for y in range(30):
        obstacles.append({"x": 0, "y": y})
        if y not in [13, 14, 15]:
            obstacles.append({"x": 49, "y": y})

    # Create bottleneck (narrow passage)
    for y in range(0, 30):
        if y not in [14, 15, 16]:
            obstacles.append({"x": 20, "y": y})
            obstacles.append({"x": 30, "y": y})
    for x in range(21, 30):
        obstacles.append({"x": x, "y": 13})

    for x in range(21, 30):
        obstacles.append({"x": x, "y": 17})

    # Generate 50 pedestrians to the left side
    pedestrians = generate_pedestrians(
        hor_span=(1, 15),
        vert_span=(6, 26),
        n_pedestrians=50,
        speed_bounds=(1.0, 1.0),
        random_seed=42,
    )
    is_absorbing = True
    distance_computation = "dijkstra"
    output_filename = os.path.join(OUTPUT_FOLDER, f"{filename}.csv")

    save_json(
        config_filename,
        grid_size,
        targets,
        measuring_points,
        obstacles,
        pedestrians,
        is_absorbing,
        distance_computation,
        output_filename,
    )


## RiMEA Test 1


def task_5_rimea_test1(filename: str):
    """Saves a configuration file for RiMEA Test 1.

    Args:
        filename (str): name for the config file
    """
    config_filename = os.path.join(CONFIG_FOLDER, f"{filename}.json")
    grid_size = {"width": 42, "height": 4}
    targets = [{"x": 41, "y": 1}]
    measuring_points = []

    # Create top and bottom walls for the corridor
    obstacles = []

    # Top wall (y=0)
    obstacles.extend(get_horizontal_object(0, 41, 0))

    # Bottom wall (y=3)
    obstacles.extend(get_horizontal_object(0, 41, 3))

    # One pedestrian with speed 1.33 m/s
    pedestrians = [{"ID": 1, "x": 1, "y": 1, "speed": 1.33}]

    is_absorbing = True
    distance_computation = "naive"
    output_filename = os.path.join(OUTPUT_FOLDER, f"{filename}.csv")

    save_json(
        config_filename,
        grid_size,
        targets,
        measuring_points,
        obstacles,
        pedestrians,
        is_absorbing,
        distance_computation,
        output_filename,
    )


## RiMEA Test 4
def rimea_test4_general(
    density: str, num_pedestrians: int, filename: str = None
):
    """Saves a configuration file for RiMEA Test 4.

    Creates a scaled-down corridor 100m long and 10m wide with three measuring
    areas of 2m x 2m as specified in the RiMEA guidelines.
    """
    if filename is None:
        filename = f"rimea_test4_{density}"

    filename = f"{filename}"
    config_filename = os.path.join(CONFIG_FOLDER, f"{filename}.json")
    output_filename = os.path.join(OUTPUT_FOLDER, f"{filename}.csv")

    # Create a small-scale version (100m × 10m)
    grid_size = {"width": 250, "height": 25}

    # Targets at the end of the corridor
    targets = []
    for y in range(5, 20):  # Center of the corridor height
        targets.append({"x": 240, "y": y})

    # Three measuring areas (2m×2m each) at 25m, 50m, and 75m
    # Each measuring area is 5x5 cells (2m×2m at 0.4m per cell)
    measuring_points = [
        {
            "ID": 1,
            "upper_left": {"x": 62, "y": 10},
            "size": {"width": 5, "height": 5},
            "delay": 0,
            "measuring_time": 1000,  # Long enough to capture steady state
        },
        {
            "ID": 2,
            "upper_left": {"x": 125, "y": 10},
            "size": {"width": 5, "height": 5},
            "delay": 0,
            "measuring_time": 1000,
        },
        {
            "ID": 3,
            "upper_left": {"x": 187, "y": 10},
            "size": {"width": 5, "height": 5},
            "delay": 0,
            "measuring_time": 1000,
        },
    ]

    # Top and bottom walls
    obstacles = []
    obstacles.extend(get_horizontal_object(0, 249, 0))  # Top wall
    obstacles.extend(get_horizontal_object(0, 249, 24))  # Bottom wall

    # Use the entire corridor width for all cases
    hor_span = (10, 230)  # Almost full corridor length (leave space at ends)
    vert_span = (2, 22)  # Full corridor height minus walls

    # Calculate available area
    available_width = hor_span[1] - hor_span[0] + 1
    available_height = vert_span[1] - vert_span[0] + 1
    available_area = available_width * available_height

    # Check if we have enough space
    if num_pedestrians > available_area:
        print(f"Warning: Not enough space for {num_pedestrians} pedestrians!")
        print(
            f"Available area: {available_area} cells, required: {num_pedestrians}"
        )
        print("Using structured placement instead of random sampling...")

        # Create pedestrians in a structured grid
        pedestrians = []
        placed = 0

        density_val = float(density.replace("_", "."))

        # Calculate spacing based on density
        spacing = max(
            1, int(0.4 / (density_val**0.5))
        )  # Adjust spacing based on density

        # Place pedestrians in a grid pattern across the entire corridor
        for y in range(vert_span[0], vert_span[1] + 1, spacing):
            for x in range(hor_span[0], hor_span[1] + 1, spacing):
                if placed < num_pedestrians:
                    pedestrians.append(
                        {"ID": placed, "x": x, "y": y, "speed": 1.33}
                    )
                    placed += 1
    else:
        # Generate pedestrians with uniform speed across the entire corridor
        pedestrians = generate_pedestrians(
            hor_span=hor_span,
            vert_span=vert_span,
            n_pedestrians=num_pedestrians,
            speed_bounds=(1.33, 1.33),  # Fixed speed of 1.33 m/s
            random_seed=42 + hash(density) % 100,  # Different seed per density
        )

    is_absorbing = True
    distance_computation = "dijkstra"  # Use dijkstra for better pathfinding

    save_json(
        config_filename,
        grid_size,
        targets,
        measuring_points,
        obstacles,
        pedestrians,
        is_absorbing,
        distance_computation,
        output_filename,
    )


def task_5_rimea_test4():
    """Generate all density variations for RiMEA Test 4 according to RiMEA guidelines

    Creates configurations with densities: 0.5, 1, 2, 3, 4, 5 and 6 pedestrians/m²
    All pedestrians have walking speeds between 1.2-1.4 m/s (using 1.33 m/s)
    """

    corridor_area = 1000  # m² (scaled down version for less memory usage)

    # Calculate number of pedestrians for each density level
    # density (ped/m²) * area (m²) = number of pedestrians
    densities = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # pedestrians/m²

    for density in densities:
        # Calculate number of pedestrians needed for this density
        num_pedestrians = int(density * corridor_area)
        density_label = f"{density:.1f}".replace(".", "_")

        print(
            f"Generating RiMEA Test 4 configuration with density {density} ped/m² "
            f"({num_pedestrians} pedestrians)"
        )

        rimea_test4_general(
            density=density_label,
            num_pedestrians=num_pedestrians,
            filename=f"rimea_test4_{density_label}ppm2",
        )


## RiMEA Test 6


def task_5_rimea_test6(filename: str):
    """Configuration for a corridor with a left turn.

    Creates a scenario where 20 pedestrians navigate a left-turning corner.
    Layout:
    - Horizontal path: 10m long * 2m wide (25*5 cells at 0.4m per cell)
    - Intersection area: 2m * 2m (5*5 cells)
    - Vertical path: 10m high * 2m wide (5*25 cells), extends upward from intersection
    Initial distribution: 20 pedestrians equally in leftmost 6m of horizontal path
    """
    config_filename = os.path.join(CONFIG_FOLDER, f"{filename}.json")

    # Grid size to accommodate the L-shaped corridor
    grid_size = {
        "width": 35,
        "height": 35,
    }  # Reduced size since we need less space

    # Target is at the top of the vertical corridor
    targets = []
    for x in range(25, 30):  # 2m wide = 5 cells, at x=25-29
        targets.append({"x": x, "y": 2})  # Near top of grid

    measuring_points = []

    # Create the L-shaped corridor walls
    obstacles = []

    # Fill the entire grid with obstacles first
    for x in range(grid_size["width"]):
        for y in range(grid_size["height"]):
            obstacles.append({"x": x, "y": y})

    # Now carve out the L-shaped corridor by removing obstacles
    removed_obstacles = []

    # Horizontal corridor (10m × 2m = 25 × 5 cells)
    for x in range(5, 30):  # 25 cells = 10m
        for y in range(20, 25):  # 5 cells = 2m
            removed_obstacles.append({"x": x, "y": y})

    # Vertical corridor (10m × 2m = 5 × 25 cells)
    for x in range(25, 30):  # 5 cells = 2m, aligned with end of horizontal
        for y in range(2, 25):  # 23 cells upward from horizontal corridor
            removed_obstacles.append({"x": x, "y": y})

    # Remove the corridor spaces from obstacles
    obstacles = [
        obs
        for obs in obstacles
        if not any(
            r["x"] == obs["x"] and r["y"] == obs["y"]
            for r in removed_obstacles
        )
    ]

    # Create 20 pedestrians uniformly distributed in the leftmost 6m of horizontal corridor
    # 6m = 15 cells at 0.4m per cell
    pedestrians = []

    # Starting area dimensions (leftmost 6m of horizontal corridor)
    start_x_min = 5  # Start of horizontal corridor
    # start_x_max = 20  # 6m = 15 cells from start
    start_y_min = 20  # Bottom of horizontal corridor
    # start_y_max = 25  # Top of horizontal corridor

    # Create 20 pedestrians in a grid pattern (4 rows × 5 columns)
    for i in range(20):
        row = i % 4  # 4 rows across corridor width
        col = i // 4  # 5 columns along corridor length

        # Calculate position with even spacing
        x = start_x_min + col * 3  # Space them out along the 6m (15 cells)
        y = start_y_min + (row + 1)  # Space them evenly across width

        pedestrians.append(
            {
                "ID": i,
                "x": x,
                "y": y,
                "speed": 1.33,  # Standard walking speed
            }
        )

    is_absorbing = True
    distance_computation = "dijkstra"  # Important for corner navigation
    output_filename = os.path.join(OUTPUT_FOLDER, f"{filename}.csv")

    save_json(
        config_filename,
        grid_size,
        targets,
        measuring_points,
        obstacles,
        pedestrians,
        is_absorbing,
        distance_computation,
        output_filename,
    )


## RiMEA Test 7
def load_data_from_csv(filepath):
    """Load age-speed mapping from CSV file."""
    age_to_speed = {}
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            age = int(float(row["age"]))
            speed = float(row["speed"])
            age_to_speed[age] = speed

    return age_to_speed


def task_5_rimea_test7(filename: str):
    """Generates a configuration file for RiMEA Test 7 (demographic parameters).

    Creates a horizontal corridor with pedestrians of different ages (and thus speeds)
    starting from the left side, moving towards a vertical target line on the right.
    """
    config_filename = os.path.join(CONFIG_FOLDER, f"{filename}.json")

    # Load age-speed data from CSV
    csv_filepath = os.path.join(CONFIG_FOLDER, "rimea_7_speeds.csv")
    age_to_speed = load_data_from_csv(csv_filepath)

    # Create a corridor-style grid with more height to fit all pedestrians
    grid_size = {"width": 100, "height": 100}  # Increased height

    # Vertical target line on the right side
    targets = []
    target_x = 100  # Place target near right side
    for y in range(0, grid_size["height"]):  # Full height for target line
        targets.append({"x": target_x, "y": y})

    # Create corridor walls along top and bottom edges
    obstacles = []
    obstacles.extend(
        get_horizontal_object(0, grid_size["width"] - 1, 0)
    )  # Top wall
    obstacles.extend(
        get_horizontal_object(
            0, grid_size["width"] - 1, grid_size["height"] - 1
        )
    )  # Bottom wall

    # Create one pedestrian for EVERY age in the CSV
    pedestrians = []
    ped_id = 0

    # Get all ages from the CSV data
    # Get all ages from the CSV data and sort them in reverse (highest age at top)
    all_ages = sorted(
        age_to_speed.keys(), reverse=True
    )  # Changed to reverse sort
    total_pedestrians = len(all_ages)
    vertical_spacing = 1
    start_y = 5
    end_y = start_y + (total_pedestrians * vertical_spacing)

    # Create vertical target line matching exactly the height of pedestrian line
    targets = []
    target_x = 95  # Place target near right side
    for y in range(start_y, end_y):  # Only span the same height as pedestrians
        targets.append({"x": target_x, "y": y})

    # Create pedestrians in a vertical line, sorted by age from top to bottom
    pedestrians = []
    for i, age in enumerate(all_ages):  # Using reverse-sorted ages
        y_pos = start_y + (i * vertical_spacing)
        speed = age_to_speed[age]

        pedestrians.append(
            {
                "ID": i,
                "x": 5,  # Fixed starting x position
                "y": y_pos,
                "speed": speed,
            }
        )
        ped_id += 1

    # Add measuring points to track progress
    measuring_points = [
        {
            "ID": i + 1,
            "upper_left": {"x": x, "y": 5},
            "size": {
                "width": 5,
                "height": grid_size["height"] - 10,
            },  # Taller measuring areas
            "delay": 0,
            "measuring_time": 1000,
        }
        for i, x in enumerate([20, 40, 60, 80])
    ]

    is_absorbing = True
    distance_computation = "dijkstra"
    output_filename = os.path.join(OUTPUT_FOLDER, f"{filename}.csv")

    save_json(
        config_filename,
        grid_size,
        targets,
        measuring_points,
        obstacles,
        pedestrians,
        is_absorbing,
        distance_computation,
        output_filename,
    )


#


if __name__ == "__main__":
    task_1("toy_example")
    task_2("task_2")

    task_4_simple("task_4_simple_obstacle")
    task_4_chicken("task_4_chicken_test")
    task_4_bottleneck("task_4_bottleneck")

    task_5_rimea_test1("task_5_rimea_test1")
    task_5_rimea_test4()
    task_5_rimea_test6("task_5_rimea_test6")
    task_5_rimea_test7("task_5_rimea_test7")

    task_5_rimea_test7("task_5_rimea_test7")
