# Prediction of Pedestrian Speed through Artificial Neural Network

## 1. Project Overview

This project aims to deeply understand and evaluate the methodology proposed in the paper by
Tordeux et al., 2019. By reproducing their core experiments, we will investigate how an Artificial Neural
Network (ANN) can be used to predict pedestrian speed, comparing its performance against a classical
Weidmann model. The report consists of five task, from diving into the paper to replicate the paper's result, until 
final performance comparison and analysis.

---

## 2. Project Setup

Follow these instructions to set up the project environment and install the necessary dependencies.

### Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) installed on your system.
- [Git](https://git-scm.com/) (For cloning the repository).

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://ge83mew@artemis.tum.de/git/MLICMS25EX6/mlicms25ex6-groupa.git
    ```

2.  **Create and activate the Conda environment:**
    ```bash
    # Create a new conda environment
    conda create --name mlcms_groupa python=3.10

    # Activate the newly created environment
    conda activate mlcms_groupa
    ```

3.  **Install dependencies:**
    Install all required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

---


## 3. File Structure

A brief overview of the key files in this project:
-   `code/`: This directory contains the source code of the project.
    -   `main.ipynb`: The core Jupyter Notebook containing the main analysis, data processing, modeling, and visualization.
    -   `utils.py`: A Python script containing helper functions used throughout the project, such as data loading utilities and preprocessing routines.
-   `README.md`: This documentation file, explaining the project and how to use it.
-   `data/`: This directory contains the raw datasets used in the analysis.
-   `requirements.txt`: A list of all Python packages required to run this project.