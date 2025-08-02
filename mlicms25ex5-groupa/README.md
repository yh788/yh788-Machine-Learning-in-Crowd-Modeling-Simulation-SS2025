# Exercise 5: Extracting dynamical systems from data

This repository contains a template for the fifth exercise. Here you are asked to implement several systems (see *src/system/*) and several models (see *src/models/*) to learn these systems. Passing the automated tests can give you up to 18 points, while the remaining 82 points are awarded manually based on the quality of your code *and* report.
The exercise sheet with the task statements: see the problem description on Artemis.

## Setup 

The repository has a folder *src/* with the source code and the folder *data/* containing required data files. You can use either Jupyter notebooks or Python scripts to run the experiments.

The task requires Python 3.11 or higher. We recommend using a [conda](https://docs.anaconda.com/free/miniconda/) or [Python virtual](https://docs.python.org/3/library/venv.html) environment to handle Python packages for the task. 

To prepare the environment, you need to:

1. Create a virtual environment of your choice and activate it.

2. Change the working directory to *mlicms24ex5-\*your_team_name\*/*.

3. Install the required packages with 

    > pip install -r requirements.txt

## Instructions
- Fill out the missing **code** (and documentation if necessary).
- You may need to define new functions to complete the task. Remember to keep the code modular and properly documented. 
- **Do not change the signatures of the existing classes and functions!** Otherwise, your code might fail some of the tests.
- Include the report in the .pdf format to the repository and submit it together with the code. 

In total, there are five tasks:
1. [task][Task 1: Approximating functions](test_data_exists,test_data_matches,test_mse,test_rbf,test_linear_solve,test_approximator_fit,test_approximator_predict)
2. [task][Task 2: Approximating linear vector fields](test_system_simulate,test_system_batch_simulate,test_system_fit,test_get_tangent,test_infer_tangent)
3. [task][Task 3: Approximating nonlinear vector fields]()
3. [task][Task 4: Time-delay embedding](test_time_delay,test_lorenz)
3. [task][Task 5: Learning crowd dynamics]()

## Linter

Having code that is compliant with a style guide is important when working in a team. There are several automatic tools that help developers to format and lint their code. In this exercise, we propose to use [*Ruff*](https://github.com/astral-sh/ruff) &mdash; a fast linter and code formatter. It is installed via a Python package in *requirements.txt*. Its parameters are specified in *ruff.toml*.

You can run *Ruff* on the source code with

> ruff check && ruff format

You should run the formatter before each commit to ensure consistent formatting across team members. 

## Git
If you are new to Git, you can read about basic Git usage in [this tutorial](https://git-scm.com/docs/gittutorial). You can also look at the [best practices](https://about.gitlab.com/topics/version-control/version-control-best-practices/) on how to work with Git to improve your workflow as a team.