import numpy as np
import numpy.typing as npt
import scipy.spatial.distance
import heapq
import math
import json
from . import elements as el
from . import utils


class Simulation:
    """
    Class for simulating the state, its initialisation, updates to the cellular automaton
    for pedestrian dynamics.

    Arguments:
    ----------
    grid_size: utils.
        The width and height of the simulation grid.
    width: int
        The width of the grid.
    height: int
        The height of the grid.
    grid: npt.NDArray [el.ScenarioElement]
        Numpy array of 2D, representing the state of each cell
    distance_grid: npt.NDArray [np.float64]
        Distance from each cell to the nearest target is stored in this grid.
    pedestrians: list [el.Pedestrian]
        Contains the Pedestrian objects in the simulation
    targets: tuple [utils.Position]
        The positions of the targets are stored in this tuple.
    distance_computation: string
        String mentioning either 'naive' or 'dijkstra' method was used.
    is_absorbing: bool
        Bool indicating whether target(s) absorb pedestrian(s) on arrival.
    output_filename: string or None
        Filename for logging simulation output.
    rng: np.random.Generator
        For shuffling, defined the random number generator.
    step_count: int
        Counter for the number of simulation steps which are performed.
    """

    def __init__(self, config: el.SimulationConfig, random_seed: int = 42):
        """Initialize simulation with configuration from JSON."""
        # Initialize basic grid properties
        self.grid_size = config.grid_size
        self.width, self.height = self.grid_size.width, self.grid_size.height
        self.grid = np.full(
            (self.width, self.height), el.ScenarioElement.empty
        )
        self.output_filename = config.output_filename
        self.distance_computation = config.distance_computation

        self.measuring_points = config.measuring_points
        self.targets = list(config.targets)
        self.pedestrians = list(config.pedestrians)
        self.obstacles = list(config.obstacles)

        # We would have to store the targets and place them on the grid.
        # prioritising pedestrians over targets then obstacles
        for pedestrian in self.pedestrians:
            if (
                0 <= pedestrian.x < self.width
                and 0 <= pedestrian.y < self.height
            ):
                self.grid[pedestrian.x, pedestrian.y] = (
                    el.ScenarioElement.pedestrian
                )
            else:
                self.pedestrians.remove(pedestrian)

        for position_target in self.targets:
            if (
                0 <= position_target.x < self.width
                and 0 <= position_target.y < self.height
            ):
                if (
                    self.grid[position_target.x, position_target.y]
                    != el.ScenarioElement.pedestrian
                ):
                    self.grid[position_target.x, position_target.y] = (
                        el.ScenarioElement.target
                    )
                else:
                    self.targets.remove(position_target)
            else:
                self.targets.remove(position_target)

        # As mention in the doc, populating grid with obstacles
        # we'll need to know where the obstacle position is

        for position_obstacle in config.obstacles:
            if (
                0 <= position_obstacle.x < self.width
                and 0 <= position_obstacle.y < self.height
            ):
                # we can't do it if cell is a target and we prefer that over obstacle
                if (
                    self.grid[position_obstacle.x, position_obstacle.y]
                    != el.ScenarioElement.pedestrian
                    and self.grid[position_obstacle.x, position_obstacle.y]
                    != el.ScenarioElement.target
                ):
                    self.grid[position_obstacle.x, position_obstacle.y] = (
                        el.ScenarioElement.obstacle
                    )
                else:
                    self.obstacles.remove(position_obstacle)
            else:
                self.obstacles.remove(position_obstacle)

        # for shuffling in _get_neighbours
        self.rng = np.random.default_rng(random_seed)
        # getting the absorbing flag for update.
        self.is_absorbing = config.is_absorbing
        self.step_count = 0
        # need to know the distance grid
        # Initialize measurement points and flows
        self.measurement_points = config.measuring_points
        self.measured_flows: dict[int, float] = {
            mp.ID: 0.0 for mp in self.measurement_points
        }

        # self.distance_grid = self._compute_distance_grid(self.targets)

    def update(self, perturb: bool = True) -> bool:
        """Performs one step of the simulation.

        Arguments:
        ----------
        perturb : bool
            If perturb=False, pedestrians' positions are updates in the
            fixed order. Otherwise, the pedestrians are shuffle before
            performing an update.

        Returns:
        --------
        bool
            True if all pedestrians reached a target and the simulation
            is over, False otherwise.
        """

        self.step_count += 1

        distances = self._compute_distance_grid(
            self.targets
        )  # Compute distance to the target for each cell
        pedestrians = list(self.pedestrians)  # Get list of pedestrians
        if perturb:  # Shuffle the list if perturb
            self.rng.shuffle(pedestrians)  # RNG shuffle

        # Initialize occupied_cells with the positions of all existing pedestrians
        occupied_cells = {
            (p.x, p.y) for p in pedestrians
        }  # Set of occupied cells to avoid collisions
        updated_positions = []  # Next state positions

        # Clear occupied_cells after speed adjustments and sorting
        occupied_cells = set()

        # Second pass: Process each pedestrian
        for pedestrian in pedestrians:
            # Ensure pedestrian has a speed attribute
            if not hasattr(
                pedestrian, "progress"
            ):  # Add a progress tracker to the pedestrian
                pedestrian.progress = 0.0  # Default progress
            pedestrian.progress += (
                pedestrian.speed
            )  # Every iteration, increase progress with speed

            current_distance = distances[
                pedestrian.x, pedestrian.y
            ]  # Get their distance to the target
            best_position = pedestrian  # Best position is current location
            best_distance = current_distance  # Best distance is current distance from current location

            # Check next possible best locations
            best_position, best_distance = self.find_next_neighbor(
                pedestrian=pedestrian,
                occupied_cells=occupied_cells,
                distances=distances,
                best_distance=best_distance,
                best_position=best_position,
            )

            # Check if best position is already occupied
            if (
                best_position.x,
                best_position.y,
            ) in occupied_cells and best_position != pedestrian:
                # Position already taken, stay in place
                updated_positions.append(pedestrian)
                continue

            # Mark the cell as occupied immediately
            occupied_cells.add((best_position.x, best_position.y))
            if (
                best_position != pedestrian
            ):  # If best position is different than initial position
                next_cell_distance = (
                    current_distance
                    - distances[best_position.x, best_position.y]
                )  # Calculate distance between best and initial position
                if (
                    next_cell_distance <= pedestrian.progress
                ):  # If distance difference smaller than progress, pedestrian can move
                    while (
                        pedestrian.progress >= next_cell_distance
                    ):  # While pedestrain can keep moving since progress is accumulated
                        current_distance = distances[
                            best_position.x, best_position.y
                        ]  # Calculate the distance of next position
                        best_position_temp, best_distance_temp = (
                            self.find_next_neighbor(  # Check next possible best locations
                                pedestrian=best_position,
                                occupied_cells=occupied_cells,
                                distances=distances,
                                best_distance=best_distance,
                                best_position=best_position,
                            )
                        )

                        # Break if at target, or best position is current position
                        if (best_position_temp.x, best_position_temp.y) == (
                            best_position.x,
                            best_position.y,
                        ) or next_cell_distance == 0:
                            break

                        # Subtract progress with each movement
                        pedestrian.progress -= next_cell_distance

                        # Break if at target, or best position is current position
                        if current_distance == 0:
                            break

                        # Calculate the next best distance
                        next_cell_distance = (
                            current_distance - best_distance_temp
                        )

                        if (
                            pedestrian.progress >= next_cell_distance
                        ):  # If pedestrian can keep moving
                            best_position = (
                                best_position_temp  # Assign best position
                            )
                            best_distance = (
                                best_distance_temp  # Assign best distance
                            )

                    self.grid[pedestrian.x, pedestrian.y] = (
                        el.ScenarioElement.empty
                    )  # Clear current position
                    # updating the grid and track absorbs
                    if (
                        self.grid[best_position.x, best_position.y]
                        == el.ScenarioElement.target
                    ):
                        if self.is_absorbing:
                            # self.grid[pedestrian.x, pedestrian.y] = el.ScenarioElement.empty
                            # self.grid[best_position.x, best_position.y] = el.ScenarioElement.pedestrian  # Mark as pedestrian
                            continue  # Absorb pedestrian (don't add to updated_positions)
                        else:
                            pedestrian.x = best_position.x  # Assign pedestrian to best position (pedestrians x coordinate)
                            pedestrian.y = best_position.y  # Assign pedestrian to best position (pedestrians y coordinate)
                            updated_positions.append(
                                pedestrian
                            )  # Stay on target
                            self.grid[best_position.x, best_position.y] = (
                                el.ScenarioElement.pedestrian
                            )  # Mark as pedestrian

                    else:
                        pedestrian.x = best_position.x  # Assign pedestrian to best position (pedestrians x coordinate)
                        pedestrian.y = best_position.y  # Assign pedestrian to best position (pedestrians y coordinate)
                        updated_positions.append(pedestrian)  # Stay on target
                        self.grid[best_position.x, best_position.y] = (
                            el.ScenarioElement.pedestrian
                        )  # Mark as pedestrian
                else:
                    updated_positions.append(
                        pedestrian
                    )  # Didn't move - add to updated positionss
            else:
                updated_positions.append(
                    pedestrian
                )  # Didn't move - add to updated positions

        # updating measuring_points
        for mp in self.measuring_points:
            # checking if its first step of the active measuring period
            if (
                self.step_count == mp.delay + 1
            ):  # +as variable is 1-indexed for measuring after 0 delay
                mp.current_speeds = []  # Clear for the new period

            # If the current step is within the (measuring) window
            if (
                self.step_count > mp.delay
                and self.step_count <= mp.delay + mp.measuring_time
            ):
                # seeing if pedestrians are in this measurement area
                for pedestrian in self.pedestrians:
                    # Check if pedestrian is inside the measurement point's area
                    is_inside_x = (
                        mp.upper_left.x
                        <= pedestrian.x
                        < mp.upper_left.x + mp.size.width
                    )
                    is_inside_y = (
                        mp.upper_left.y
                        <= pedestrian.y
                        < mp.upper_left.y + mp.size.height
                    )

                    if is_inside_x and is_inside_y:
                        mp.current_speeds.append(pedestrian.speed)

        self.pedestrians = (
            updated_positions  # Assign next state of pedestrians to grid
        )
        # Return True if all pedestrians have reached the target
        return len(self.pedestrians) == 0

    def find_next_neighbor(
        self,
        pedestrian,
        occupied_cells,
        distances,
        best_distance,
        best_position,
    ) -> tuple[utils.Position, float]:
        """
        The function computes the best possible next position and the distance w.r.t to the given pedestrian cell as the argument.

        Arguments:
        ----------
        pedestrian : utils.Position
            Target pedestrian to find its neighbors and return the best one.
        occupied_cells : set[tuple[int, int]]
            A set of grid coordinates already occupied, to avoid collisions.
        distances : npt.NDArray[np.float64]
            2D NumPy array representing precomputed shortest distances from all cells to the target.
        best_distance : float
            The current best, minimum, distance found so far for the target pedestrian.
        best_position : utils.Position
            The current best cell position based on minimum distance towards the target. Updated if a better neighbor is found.

        Returns:
        --------
        Tuple[utils.Position, float]
            Returns the neighboring cell with the lowest distance to the target and it's distance from all possible neighbors
        """

        for neighbor in self._get_neighbors(pedestrian):  # For each neighbor
            # Skip cells that are already occupied by other pedestrians
            if (neighbor.x, neighbor.y) in occupied_cells:
                continue
            # Avoid non-empty cells (except targets)
            if self.grid[neighbor.x, neighbor.y] not in [
                el.ScenarioElement.empty,
                el.ScenarioElement.target,
            ]:
                continue

            neighbor_distance = distances[
                neighbor.x, neighbor.y
            ]  # Check the distance of that cell to the target
            if neighbor_distance < best_distance:  # If distance is smaller
                # Check if this is a target cell that's already been claimed
                if (
                    self.grid[neighbor.x, neighbor.y]
                    == el.ScenarioElement.target
                    and (neighbor.x, neighbor.y) in occupied_cells
                ):
                    continue  # Skip this target cell if already claimed

                best_position = neighbor  # Assign best position to that cell
                best_distance = (
                    neighbor_distance  # Assign best distance to that cell
                )

        return (
            best_position,
            best_distance,
        )  # Return best neighboring position and distance

    def get_grid(self) -> npt.NDArray[el.ScenarioElement]:
        """Returns a full state grid of the shape (width, height)."""

        # TODO: return a grid for visualization...DONE!
        # DONE...just have to use np.copy
        grid = np.copy(self.grid)
        return grid

    def get_distance_grid(self) -> npt.NDArray[np.float64]:
        """Returns a grid with distances to a closest target."""
        # TODO: return a distance grid...DONE
        distance_grid = self._compute_distance_grid(self.targets)
        return distance_grid

    def get_measured_flows(self) -> dict[int, float]:
        """Returns a map of measuring points' ids to their flows.

        Returns:
        --------
        dict[int, float]
            A dict in the form {measuring_point_id: flow}.
        """
        return {mp.ID: mp.get_mean_flow() for mp in self.measuring_points}

    def _compute_distance_grid(
        self, targets: tuple[utils.Position]
    ) -> npt.NDArray[np.float64]:
        """
        The function computes the distance w.r.t each cell to the nearest target.

        The computation method can be 'naive' or 'dijkstra', governed by 'self.distance_computation'.

        Arguments:
        ----------
        targets : tuple[utils.Position, ...]
            Target positions on the grid given by this tuple .

        Returns:
        --------
        npt.NDArray[np.float64]
            An array of distances having the same shape as the main grid.
            Cells which are unreachable, will have infinite (np.inf) distance if using 'dijkstra' method.
        """

        if len(targets) == 0:
            distances = np.zeros((self.width, self.height))
            return distances

        match self.distance_computation:
            case "naive":
                distances = self._compute_naive_distance_grid(targets)
            case "dijkstra":
                distances = self._compute_dijkstra_distance_grid(targets)
            case _:
                print(
                    "Unknown algorithm for computing the distance grid: "
                    f"{self.distance_computation}. Defaulting to the "
                    "'naive' option."
                )
                distances = self._compute_naive_distance_grid(targets)
        return distances

    def _compute_naive_distance_grid(
        self, targets: tuple[utils.Position]
    ) -> npt.NDArray[np.float64]:
        """Computes a distance grid without considering obstacles.

        Arguments:
        ----------
        targets : Tuple[utils.Position]
            A tuple of targets on the grid. For each cell, the algorithm
            computes the distance to the closest target.

        Returns:
        --------
        npt.NDArray[np.float64]
            An array of distances of the same shape as the main grid.
        """

        targets = [[*target] for target in targets]
        targets = np.vstack(targets)
        x_space = np.arange(0, self.width)
        y_space = np.arange(0, self.height)
        xx, yy = np.meshgrid(x_space, y_space)
        positions = np.column_stack([xx.ravel(), yy.ravel()])

        # after the target positions and all grid cell positions are stored,
        # compute the pair-wise distances in one step with scipy.
        distances = scipy.spatial.distance.cdist(targets, positions)

        # now, compute the minimum over all distances to all targets.
        distances = np.min(distances, axis=0)
        distances = distances.reshape((self.height, self.width)).T

        return distances

    def _compute_dijkstra_distance_grid(
        self, targets: tuple[utils.Position]
    ) -> npt.NDArray[np.float64]:
        """Computes a distance grid considering obstacles using dijkstra algorithm.

        Arguments:
        ----------
        targets : Tuple[utils.Position]
            A tuple of targets on the grid. For each cell, the algorithm
            computes the distance to the nearest target.

        Returns:
        --------
        npt.NDArray[np.float64]
            An array of the same shape as the main grid, representing the minimal distance
            to the nearest target. Unreachable cells have infinite distance value.
        """
        # Initialize returned array.
        distances = np.full((self.width, self.height), np.inf)
        # Initialize priority queue, which includes all to be expanded cells.
        priority_queue = []
        # Define target's distance to the targets as 0 and push all targets in the queue
        for target in targets:
            # ensuring targets are within bounds
            if 0 <= target.x < self.width and 0 <= target.y < self.height:
                x, y = target.x, target.y
                distances[x, y] = 0
                heapq.heappush(priority_queue, (0, (x, y)))

        # Begin of main loop, exit when all cells are popped out of priority queue.
        while priority_queue:
            distance, (x, y) = heapq.heappop(priority_queue)

            # Check if current distance of the cell is the smallest distance: if not, skip this cell.
            if distance > distances[x, y]:
                continue

            # Expand the cell by getting the neighbors.
            neighbors = self._get_neighbors(utils.Position(x, y))

            for neighbor in neighbors:
                n_x, n_y = neighbor.x, neighbor.y

                # Ignore obstacle cells.
                if self.grid[n_x, n_y] == el.ScenarioElement.obstacle:
                    continue

                # Choose between orthogonal and diagonal movement.
                if abs(n_x - x) + abs(n_y - y) == 1:
                    n_distance = distance + 1
                elif abs(n_x - x) and abs(n_y - y) == 1:
                    n_distance = distance + math.sqrt(2)

                # Update neighboring cells' value, only if it is smallest among all distance values of this cell.
                if n_distance < distances[n_x, n_y]:
                    distances[n_x, n_y] = n_distance
                    heapq.heappush(priority_queue, (n_distance, (n_x, n_y)))

        return distances

    def _get_neighbors(
        self, position: utils.Position, shuffle: bool = True
    ) -> list[utils.Position]:
        """Returns a list of neighboring cells for the position.

        Arguments:
        ----------
        positions : utils.Position
            A position on the grid.
        shuffle : bool
            An indicator if neighbors should be shuffled or returned
            in the fixed order.

        Returns:
        --------
        list[utils.Position]
            An array of neighboring cells. Two cells are neighbors
            if they share a common vertex.
        """

        neighbors = []
        pos_x = position.x
        pos_y = position.y
        # have to iterate over 3 by 3 neighbourhood including the centre,
        # which is to be ignored
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                if x == y == 0:
                    continue
                neighbors_x = pos_x + x
                neighbors_y = pos_y + y
                # checking the bounds
                if (
                    0 <= neighbors_x < self.width
                    and 0 <= neighbors_y < self.height
                ):
                    neighbors.append(utils.Position(neighbors_x, neighbors_y))
        if shuffle:
            self.rng.shuffle(neighbors)
        return neighbors

    def _post_process(self):
        """Post-processing steps after simulation like saving data and such"""

        if self.output_filename is None:
            return

        with open(self.output_filename, "w", newline="") as file:
            data_written = {
                "grid": self.grid.tolist(),
                "pedestrians": [
                    {
                        "ID": getattr(pedestrian, "ID", None),
                        "x": pedestrian.x,
                        "y": pedestrian.y,
                        "speed": getattr(pedestrian, "speed", 1.0),
                        "age": getattr(pedestrian, "age", None),
                    }
                    for pedestrian in self.pedestrians
                ],
            }
            json.dump(data_written, file, indent=4)
        # TODO: store output for analysis.
        print(f"Simulation complete. Output saved to {self.output_filename}.")
