# Exercise 1: Cellular Automaton 

This repository contains a template for the first exercise. Here you are asked to implement a simple cellular automaton and test it in different scenarios. The exercise consists of two parts: implementing the automaton and writing a detailed report on it. Passing the automated tests can give you up to 25 points, while the remaining 75 points are awarded manually based on the quality of your code *and* report.

The exercise sheet with the task statements: (TODO: include the pdf in the Artemis' problem description).

## Setup 

The repository has three folders: *src/* with the source code, *configs/* with configuration files for the GUI and simulations, and *outputs/* with the output statistics of simulations.

The task requires Python 3.11 or higher. We recommend using a [conda](https://docs.anaconda.com/free/miniconda/) or [Python virtual](https://docs.python.org/3/library/venv.html) environment to handle Python packages for the task. 

To run the GUI, you need to:

1. Create a virtual environment of your choice and activate it.

2. Change the working directory to *mlicms24ex1-\*your_team_name\*/*.

3. Install the required packages with 

    > pip install -r requirements.txt

4. Create a configuration file manually or (preferably) generate it with

    > python src/generate_configs.py

5. Start the GUI with

    > python main.py --gui configs/gui.json --simulation configs/\*config_name\*.json

## Instructions
- Fill out the missing **code and documentation** in the *src/* folder to implement the cellular automaton. 
- You can define new classes and functions to complete the task. But don't forget to keep the code modular and properly documented.
- **Do not change the signatures of the existing classes and functions!** Otherwise, your code might fail some of the tests.
- Include the report in the .pdf format to the repository and submit it together with the code. 

In total, there are five tasks:
1. [task][Task 1: setting up](test_normal_init,test_overlapping_init,test_rectangular_grid,test_empty_grid)
2. [task][Task 2: first steps](test_center,test_edges,test_corners,test_straight_line,test_update_speed)
3. [task][Task 3: pedestrian interaction](test_absorbs,test_not_absorbs,test_no_overstepping,test_no_target_together,test_diagonals,test_speeds)
3. [task][Task 4: obstacle avoidance](test_no_crossing,test_unreachable,test_no_obstacles,test_with_obstacles,test_edge_obstacle,test_several_targets,test_with_pedestrians,test_dijkstra_speed)
4. [task][Task 5: RiMEA tests](test_delay,test_measuring_time,test_large_area,test_speed,test_reached,test_two_points,test_two_pedestrians,test_not_tracked)

## Linter

Having code that is compliant with a style guide is important when working in a team. There are several automatic tools that help developers to format and lint their code. In this exercise, we propose to use [*Ruff*](https://github.com/astral-sh/ruff) &mdash; a fast linter and code formatter. It is installed via a Python package in *requirements.txt*. Its parameters are specified in *ruff.toml*.

You can run *Ruff* on the source code with

> ruff check . && ruff format .

You should run the formatter before each commit to ensure consistent formatting across team members. 

## Git
If you are new to Git, you can read about basic Git usage in [this tutorial](https://git-scm.com/docs/gittutorial). You can also look at the [best practices](https://about.gitlab.com/topics/version-control/version-control-best-practices/) on how to work with Git to improve your workflow as a team.