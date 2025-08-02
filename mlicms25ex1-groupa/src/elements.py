from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

import numpy as np
from . import utils


class ScenarioElement(np.int64, Enum):
    """A class to represent a state of the cell."""

    empty = 0
    target = 1
    obstacle = 2
    pedestrian = 3


@dataclass
class Pedestrian(utils.Position):
    """
    A class to represent a single pedestrian.

    Attributes:
    -----------
    ID : int
        Unique ID of the pedestrian.
    x : int
        The x coordinate of the pedestrian's current position.
    y : int
        The y coordinate of the pedestrian's current position.
    position : utils.Position
        The (x, y) pedestrian's position.
    speed : float
        Pedestrian's speed in the units of cells/step.
    took_steps : int | None = None
        The number of steps that the pedestrian took to reach the
        target. If took_steps is None, then the pedestrian hasn't
        reached the target yet.
    """

    ID: int
    x: int
    y: int
    speed: float
    took_steps: int | None = None

    def get_position(self):
        return utils.Position(self.x, self.y)

    def set_position(self, position):
        self.x, self.y = position.x, position.y

    position = property(get_position, set_position)


@dataclass
class MeasuringPoint:
    """
    A class to represent a rectangular measuring point.

    A measuring point records pedestrians in its measuring area at the
    beginning of a step and stores their density with respect to the
    measuring area. After the step, the point stores the actual speed
    of recorded pedestrians. Using the stored values, it can then
    compute the mean measured flow.

    Attributes:
    -----------
    ID : int
        Unique ID of the measuring point.
    upper_left : utils.Position
        The coordinate of the measuring rectangle's upper left corner.
    size : utils.Size
        The size of the measuring rectangle.
    delay : int
        Number of steps to wait before start of the measuring.
    measuring_time : int
        Number of steps in the measuring period.
    current_speeds : list[float]
        Speeds of all pedestrians in the measurement area

    Methods:
    --------
    get_mean_flow() -> np.float64:
        Computes the mean pedestrian flow across the measuring period.
    """

    ID: int
    upper_left: utils.Position
    size: utils.Size
    delay: int
    measuring_time: int
    current_speeds: list[float]

    @classmethod
    def from_dict(cls, config_dict: dict):
        upper_left = utils.Position(**config_dict["upper_left"])
        size = utils.Size(**config_dict["size"])
        return cls(
            ID=config_dict["ID"],
            upper_left=upper_left,
            size=size,
            delay=config_dict["delay"],
            measuring_time=config_dict["measuring_time"],
            current_speeds=[],
        )

    def get_mean_flow(self) -> np.float64:
        area = self.size.height * self.size.width
        if area == 0 or len(self.current_speeds) == 0:
            return np.float64(0.0)
        mean_speed = np.mean(self.current_speeds)
        return np.float64(mean_speed / area)


@dataclass
class Colors:
    """A class to specify RGB colors for each ScenarioElement."""

    empty: tuple[int, int, int]
    target: tuple[int, int, int]
    obstacle: tuple[int, int, int]
    pedestrian: tuple[int, int, int]

    def __getitem__(self, element: ScenarioElement):
        match element:
            case ScenarioElement.empty:
                return self.empty
            case ScenarioElement.target:
                return self.target
            case ScenarioElement.obstacle:
                return self.obstacle
            case ScenarioElement.pedestrian:
                return self.pedestrian
            case _:
                print(f"Received an unknown scenario elements: {element}.")
                return [0, 0, 0]


@dataclass
class GUIConfig:
    """The configuration parameters of the GUI.

    Attributes:
    -----------
    colors : Colors
        The colors of each ScenarioElement.
    window_size : utils.Size
        The size of the main window in pixels.
    button_height : int
        The height of the button area in pixels.
    step_ms : int
        The time in ms to wait between two successive steps of the
        simulation.
    """

    colors: Colors
    window_size: utils.Size
    buttons_height: int
    step_ms: int

    @classmethod
    def from_dict(cls, config_dict: dict):
        colors = Colors(**config_dict["colors"])
        window_size = utils.Size(**config_dict["window_size"])
        return cls(
            colors=colors,
            window_size=window_size,
            buttons_height=config_dict["buttons_height"],
            step_ms=config_dict["step_ms"],
        )


@dataclass
class SimulationConfig:
    """The configuration parameters of the simulation.

    Attributes:
    -----------
    grid_size : utils.Size
        The size of the grid in cells.
    targets : tuple[utils.Position]
        A tuple of the target's positions on the grid.
    obstacles : tuple[utils.Position]
        A tuple of the obstacle's positions on the grid.
    measuring_points : list[MeasuringPoint]
        A list of measuring points in the simulation.
    pedestrians : list[Pedestrian]
        A list of pedestrians in their starting positions.
    is_absorbing : bool
        An indicator if targets should absorb the pedestrians.
        If a target is absorbing, then a pedestrian that is about
        to reach a target:
            1) Steps on the target cell.
            2) Stays there for the rest of the step.
            3) Is removed from the simulation on the next step when
            their turn to move comes.
    distance_computation : str = "naive"
        The name of the algorithm for distance computation.
    output_filename : str = "output.csv"
        The .csv file for storing the output of the simulation.
        If set to None, the output will not be logged.
    """

    grid_size: utils.Size
    targets: tuple[utils.Position]
    obstacles: tuple[utils.Position]
    measuring_points: list[MeasuringPoint]
    pedestrians: list[Pedestrian]
    is_absorbing: bool = True
    distance_computation: str = "naive"
    output_filename: str | None = None

    @classmethod
    def from_dict(cls, config_dict):
        grid_size = utils.Size(**config_dict["grid_size"])
        targets = tuple(
            utils.Position(**target) for target in config_dict["targets"]
        )
        obstacles = tuple(
            utils.Position(**obstacle) for obstacle in config_dict["obstacles"]
        )
        pedestrians = [
            Pedestrian(**pedestrian)
            for pedestrian in config_dict["pedestrians"]
        ]

        measuring_points = [
            MeasuringPoint.from_dict(point_dict)
            for point_dict in config_dict["measuring_points"]
        ]
        return cls(
            grid_size=grid_size,
            targets=targets,
            obstacles=obstacles,
            measuring_points=measuring_points,
            pedestrians=pedestrians,
            is_absorbing=config_dict["is_absorbing"],
            output_filename=config_dict["output_filename"],
            distance_computation=config_dict["distance_computation"],
        )

    def __post_init__(self):
        self.targets = self._filter_invalid(self.targets, label="target")
        self.obstacles = self._filter_invalid(self.obstacles, label="obstacle")
        self.pedestrians = self._filter_invalid(
            self.pedestrians, label="pedestrian"
        )

    def _is_inside(self, element: utils.Position) -> bool:
        """Checks if the given element is inside the grid."""
        valid_x = 0 <= element.x < self.grid_size.width
        valid_y = 0 <= element.y < self.grid_size.height
        return valid_x and valid_y

    def _filter_invalid(
        self, elements: Iterable[utils.Position], label: str = "coordinate"
    ) -> list[utils.Position]:
        """Filters out elements outside the grid and prints a warning.

        Arguments:
        ----------
        elements : List[utils.Position]
            A list of elements to filter.
        label : str = 'coordinate'
            A label to use in the warning message.
        """
        valid = []
        for element in elements:
            if not self._is_inside(element):
                print(
                    f"WARNING: the {label} {element} "
                    f"is invalid for the grid size {self.grid_size}."
                )
            else:
                valid.append(element)
        return valid
