import os
import matplotlib.pyplot as plt
import csv

from src.utils import parse_json  # Only import parse_json
from src.simulation import Simulation


def dict_to_obj(dictionary):
    """Convert a dictionary to an object with attributes for Simulation class."""

    class Config:
        def __iter__(self):
            if hasattr(self, "x") and hasattr(self, "y"):
                return iter([self.x, self.y])
            raise TypeError("This Config object is not iterable")

        def __str__(self):
            return str(vars(self))

    def convert_dict(d):
        obj = Config()
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(obj, key, convert_dict(value))
            elif isinstance(value, list):
                # Convert list items if they're dictionaries
                obj_list = []
                for item in value:
                    if isinstance(item, dict):
                        obj_list.append(convert_dict(item))
                    else:
                        obj_list.append(item)
                setattr(obj, key, obj_list)
            else:
                setattr(obj, key, value)
        return obj

    return convert_dict(dictionary)


def run_rimea_test1():
    """
    Run RiMEA Test 1: Straight Line Movement
    Tests if a pedestrian can maintain specified walking speed (1.33 m/s)
    """
    print("Running RiMEA Test 1: Straight Line Movement")

    # Use parse_json to get a dictionary, then convert to object
    config_file = "configs/task_5_rimea_test1.json"
    try:
        config_dict = parse_json(config_file)
        print(f"Successfully loaded configuration from {config_file}")

        # Convert dictionary to object with attributes as Simulation class doesn't accept dict
        config = dict_to_obj(config_dict)
        print(
            "Successfully converted config dictionary to object with attributes"
        )

    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    sim = Simulation(config)

    # Track steps and positions
    steps = 0
    positions = []
    max_steps = (
        50  # Arbitrary limit (max steps shoul be 40 for 40m with speed 1 m/s)
    )

    # Run simulation until pedestrian reaches target or max steps
    while len(sim.pedestrians) > 0 and steps < max_steps:
        # Record positions
        if sim.pedestrians:
            positions.append((sim.pedestrians[0].x, sim.pedestrians[0].y))

        # Update simulation
        sim.update(
            perturb=False
        )  # No random perturbation for reproducible results
        steps += 1

    # RiMEA Test 1 should complete in 26-34 seconds for 40m at 1.33 m/s
    print(f"Test completed in {steps} steps")

    # Check if within expected range
    if 26 <= steps <= 34:
        print("PASSED: Travel time within expected range (26-34 steps)")
    else:
        print(
            f"FAILED: Travel time ({steps}) outside expected range (26-34 steps)"
        )

    return 0


def run_rimea_test4():
    """This function runs RiMEA Test 4: Fundamental Diagram (Density vs Speed)
    Tests the relationship between pedestrian density and walking speed.
    """
    print("Running RiMEA Test 4: Fundamental Diagram")

    # Updated density labels to match the new config file naming
    densities = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # Actual density values
    density_labels = [
        "0_5",
        "1_0",
        "2_0",
        "3_0",
        "4_0",
        "5_0",
        "6_0",
    ]  # Config file labels
    results = {}

    for i, density_label in enumerate(density_labels):
        # Updated config filename pattern to match the new files
        config_file = f"configs/rimea_test4_{density_label}ppm2.json"
        print(
            f"\nRunning density level: {densities[i]} ped/m² (density label: {density_label})"
        )

        try:
            # Load the configuration
            config_dict = parse_json(config_file)
            print(f"Successfully loaded configuration from {config_file}")

            # Convert dictionary to object with attributes
            config = dict_to_obj(config_dict)
            print(
                "Successfully converted config dictionary to object with attributes"
            )
        except FileNotFoundError:
            print(
                f"Error: Config file {config_file} not found. Skipping this density level."
            )
            continue
        except Exception as e:
            print(f"Error loading configuration: {e}")
            continue

        try:
            # Start simulation
            sim = Simulation(config)

            # Run for 70 steps - 10 seconds transient + 60 seconds measurement
            steps = 0
            max_steps = 70

            # Lists to store speed measurements
            speed_measurements = []

            # Find measuring points
            measuring_points = []
            if hasattr(config, "measuring_points"):
                measuring_points = config.measuring_points

            # Group measuring points by x coordinate
            measuring_areas = {}
            for mp in measuring_points:
                try:
                    if hasattr(mp, "upper_left") and hasattr(
                        mp.upper_left, "x"
                    ):
                        x = mp.upper_left.x
                        if x not in measuring_areas:
                            measuring_areas[x] = []
                        measuring_areas[x].append(mp)
                except Exception as e:
                    print(f"Error processing measuring point: {e}")
                    print(f"Measuring point structure: {mp}")
                    continue

            # Run simulation
            speed = 0
            measured_peds = {x: set() for x in measuring_areas.keys()}
            while len(sim.pedestrians) > 0 and steps < max_steps:
                if steps > 10:  # Skip first 10 seconds
                    for ped in sim.pedestrians:
                        for area_x, points in measuring_areas.items():
                            # Only measure each pedestrian once per measuring point
                            if ped.ID in measured_peds[area_x]:
                                continue
                            for mp in points:
                                # Check if pedestrian is within measuring area bounds
                                if (
                                    mp.upper_left.x
                                    <= ped.x
                                    < mp.upper_left.x + mp.size.width
                                    and mp.upper_left.y
                                    <= ped.y
                                    < mp.upper_left.y + mp.size.height
                                ):
                                    # The speed is already in m/s, no need to multiply by 0.4
                                    speed_measurements.append(ped.speed)
                                    speed = +ped.speed * 0.4

                                    break

                sim.update(perturb=False)
                steps += 1

                # Progress indicator
                if steps % 20 == 0:
                    print(
                        f"Step {steps}, remaining pedestrians: {len(sim.pedestrians)}"
                    )

            # Calculate average speed from all measurements
            avg_speed = (
                speed / len(speed_measurements)
                if len(speed_measurements) > 0
                else 0
            )
            # Store results
            results[densities[i]] = {
                "speed": avg_speed,
                "flow": avg_speed * densities[i],  # flow = speed * density
                "num_measurements": len(speed_measurements),
            }

            print(
                f"Density: {densities[i]:.1f} ped/m², Average Speed: {avg_speed:.4f} m/s"
            )
            print(f"Number of speed measurements: {len(speed_measurements)}")

        except Exception as e:
            print(f"Error running simulation: {e}")

    # Plot the results if we have any
    if results:
        # Sort densities for plotting
        densities_values = sorted(results.keys())

        # Extract speed and flow values
        speeds = [results[d]["speed"] for d in densities_values]
        flows = [results[d]["flow"] for d in densities_values]

        # Plot density vs speed (the main requirement)
        plt.figure(figsize=(10, 6))
        plt.plot(densities_values, speeds, "ro-", linewidth=2)
        plt.scatter(densities_values, speeds, color="red", s=100)

        # Add data point labels
        for i, (d, s) in enumerate(zip(densities_values, speeds)):
            plt.annotate(
                f"{d:.1f} ped/m²\n({s:.3f} m/s)",
                (d, s),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        plt.grid(True)
        plt.title("RiMEA Test 4: Density vs Speed", fontsize=16)
        plt.xlabel("Density (pedestrians/m²)", fontsize=14)
        plt.ylabel("Speed (m/s)", fontsize=14)
        plt.savefig("rimea_test4_speed_vs_density.png")
        plt.show()

        # Fflow diagram for reference
        plt.figure(figsize=(10, 6))
        plt.plot(densities_values, flows, "bo-", linewidth=2)
        plt.scatter(densities_values, flows, color="blue", s=100)
        plt.grid(True)
        plt.title("RiMEA Test 4: Density vs Flow", fontsize=16)
        plt.xlabel("Density (pedestrians/m²)", fontsize=14)
        plt.ylabel("Flow (pedestrians/m/s)", fontsize=14)
        plt.savefig("rimea_test4_flow_vs_density.png")
        plt.show()
    else:
        print("No results collected to plot.")

    return 0


def get_line_points(x1: int, y1: int, x2: int, y2: int) -> list:
    """Get all points along a line using Bresenham's line algorithm

    Returns:
        _type_: list of tuples representing points along the line
    """
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1

    if dx > dy:
        err = dx / 2.0
        while x != x2:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y2:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    points.append((x2, y2))
    return points


def check_wall_violation(
    prev_x: int, prev_y: int, curr_x: int, curr_y: int, obstacles: list
) -> bool:
    """Check if movement between two points crosses through any walls.

    Args:
        prev_x (int): previous x-coordinate
        prev_y (int): previous y-coordinate
        curr_x (int): current x-coordinate
        curr_y (int):  current y-coordinate
        obstacles (list): list of obstacle objects with x and y attributes

    Returns:
        bool: True if movement crosses through walls, False otherwise
    """

    # Convert obstacles to set of tuples for faster lookup
    obstacle_set = {(obs.x, obs.y) for obs in obstacles}

    # Get all points along the movement path
    path_points = get_line_points(prev_x, prev_y, curr_x, curr_y)

    # Check if any point in the path intersects with obstacles
    return any(point in obstacle_set for point in path_points)


def run_rimea_test6():
    """
    Run RiMEA Test 6: Movement around a corner

    Tests if 20 pedestrians can successfully navigate a left-turning corner
    without passing through walls. The corridor consists of two perpendicular
    segments, each 10m long and 2m wide.
    """
    print("Running RiMEA Test 6: Movement around a corner")

    # Use parse_json to get a dictionary, then convert to object
    config_file = "configs/task_5_rimea_test6.json"
    try:
        config_dict = parse_json(config_file)
        print(f"Successfully loaded configuration from {config_file}")

        # Convert dictionary to object with attributes
        config = dict_to_obj(config_dict)
        print(
            "Successfully converted config dictionary to object with attributes"
        )

    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    try:
        # Start simulation
        sim = Simulation(config)

        # Track steps and pedestrian counts
        steps = 0
        max_steps = (
            200  # Allow enough time for all pedestrians to navigate the corner
        )

        # Store pedestrian counts at each step
        ped_counts = []
        corner_positions = set()  # Track unique positions in the corner area

        # Define corner area (the intersection of horizontal and vertical corridors)
        corner_x_min, corner_x_max = 17, 22
        corner_y_min, corner_y_max = 20, 25

        # Track if any pedestrian passes through walls
        wall_violations = 0
        prev_positions = {}  # Track previous positions to detect wall crossing

        print(f"Starting simulation with {len(sim.pedestrians)} pedestrians")

        # Run simulation until all pedestrians reach targets or max steps
        while len(sim.pedestrians) > 0 and steps < max_steps:
            # Record pedestrian count
            ped_counts.append(len(sim.pedestrians))

            # Check for corner navigation and wall violations
            for ped in sim.pedestrians:
                # Record positions in corner area
                if (
                    corner_x_min <= ped.x <= corner_x_max
                    and corner_y_min <= ped.y <= corner_y_max
                ):
                    corner_positions.add((ped.x, ped.y))

                # Check for wall violations (if we have previous position)
                if hasattr(ped, "ID") and ped.ID in prev_positions:
                    prev_x, prev_y = prev_positions[ped.ID]
                    curr_x, curr_y = ped.x, ped.y

                    # Check for actual wall violations using obstacle data
                    if check_wall_violation(
                        prev_x, prev_y, curr_x, curr_y, config.obstacles
                    ):
                        print(
                            f"WARNING: Pedestrian {ped.ID} violated wall constraint: "
                            f"({prev_x},{prev_y}) -> ({curr_x},{curr_y})"
                        )
                        wall_violations += 1

                # Update previous positions
                if hasattr(ped, "ID"):
                    prev_positions[ped.ID] = (ped.x, ped.y)

            # Update simulation
            sim.update(perturb=False)
            steps += 1

            # Progress indicator
            if steps % 20 == 0:
                print(
                    f"Step {steps}, remaining pedestrians: {len(sim.pedestrians)}"
                )

        # Analyze results
        print(f"\nSimulation completed in {steps} steps")
        print(
            f"Pedestrians that reached the target: {20 - len(sim.pedestrians)} out of 20"
        )
        print(f"Unique positions used in corner area: {len(corner_positions)}")
        print(f"Wall violations detected: {wall_violations}")

        # Determine test result
        if (
            wall_violations == 0 and (20 - len(sim.pedestrians)) >= 18
        ):  # Allow for small issues
            print(
                "PASSED: Pedestrians successfully navigated the corner without wall violations"
            )
        else:
            print("FAILED: Issues detected in corner navigation")
            if wall_violations > 0:
                print(f"- {wall_violations} potential wall violations")
            if (20 - len(sim.pedestrians)) < 18:
                print(
                    f"- Only {20 - len(sim.pedestrians)} pedestrians reached the target"
                )

        # Plot pedestrian count over time
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(ped_counts)), ped_counts, "b-", linewidth=2)
        plt.grid(True)
        plt.title("RiMEA Test 6: Movement around a corner", fontsize=16)
        plt.xlabel("Simulation step", fontsize=14)
        plt.ylabel("Remaining pedestrians", fontsize=14)
        plt.savefig("rimea_test6_results.png")
        plt.show()

        return 0

    except Exception as e:
        print(f"Error running simulation: {e}")
        import traceback

        traceback.print_exc()
        return 1


## Test 7


def run_rimea_test7():
    """
    Run RiMEA Test 7: Age-based movement speeds
    Visualizes:
    1. Age vs Speed from CSV data
    2. Age vs Time taken to reach target from simulation
    """
    print("Running RiMEA Test 7: Age-based speeds")

    # Load age-speed data from CSV
    csv_file = "configs/rimea_7_speeds.csv"
    ages = []
    speeds = []
    try:
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ages.append(int(float(row["age"])))
                speeds.append(float(row["speed"]))
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return 1

    # Load and run simulation
    config_file = "configs/task_5_rimea_test7.json"
    try:
        config_dict = parse_json(config_file)
        config = dict_to_obj(config_dict)
        sim = Simulation(config)

        # Track time taken for each pedestrian to reach target
        completion_times = {}  # {pedestrian_id: steps_taken}
        active_pedestrians = {ped.ID: ped for ped in sim.pedestrians}
        steps = 0
        max_steps = 200

        print(
            f"Starting simulation with {len(active_pedestrians)} pedestrians"
        )

        while len(sim.pedestrians) > 0 and steps < max_steps:
            # Check which pedestrians reached target
            for ped_id in list(active_pedestrians.keys()):
                if ped_id not in [p.ID for p in sim.pedestrians]:
                    completion_times[ped_id] = steps
                    del active_pedestrians[ped_id]

            sim.update(perturb=False)
            steps += 1

            if steps % 20 == 0:
                print(
                    f"Step {steps}, remaining pedestrians: {len(sim.pedestrians)}"
                )

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Age vs Speed from CSV
        ax1.plot(ages, speeds, "bo-")
        ax1.set_title("Age vs Speed (from CSV)")
        ax1.set_xlabel("Age (years)")
        ax1.set_ylabel("Speed (m/s)")
        ax1.grid(True)

        # Plot 2: Age vs Time to Target
        completion_times_list = []
        for age, ped_id in zip(ages, range(len(ages))):
            if ped_id in completion_times:
                completion_times_list.append((age, completion_times[ped_id]))

        if completion_times_list:
            plot_ages, plot_times = zip(*sorted(completion_times_list))
            ax2.plot(plot_ages, plot_times, "ro-")
            ax2.set_title("Age vs Time to Target")
            ax2.set_xlabel("Age (years)")
            ax2.set_ylabel("Time Steps")
            ax2.grid(True)

        plt.tight_layout()
        plt.savefig("rimea_test7_results.png")
        plt.show()

        # Print statistics
        print("\nSimulation Results:")
        print(f"Total pedestrians: {len(ages)}")
        print(f"Pedestrians that reached target: {len(completion_times)}")
        print(
            f"Average completion time: {sum(completion_times.values()) / len(completion_times):.2f} steps"
        )

    except Exception as e:
        print(f"Error running simulation: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    if not os.path.exists("configs/task_7_rimea_test1.json"):
        print("Error: RiMEA configuration files not found.")

    run_rimea_test1()
    run_rimea_test4()
    run_rimea_test6()
    run_rimea_test7()
