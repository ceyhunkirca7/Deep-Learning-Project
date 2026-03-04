# Deep Learning Image Processing: MLP & CNN Implementations

This project contains 5 different Python scripts demonstrating the use of Multi-Layer Perceptron (MLP) and Convolutional Neural Networks (CNN) using PyTorch. Each script generates its own synthetic dataset of 25x25 matrices, trains a model to solve a specific distance or counting problem, and evaluates the performance visually.

## Project Contents

The original files have been renamed to reflect their functionalities in English:

1. **`mlp_distance_between_two_points.py`** 
   - **Model:** MLP (Multi-Layer Perceptron)
   - **Task:** Generates a 25x25 matrix with exactly 2 points (represented by 1s) and uses an MLP to predict the Euclidean distance between these two points.

2. **`cnn_shortest_distance.py`** 
   - **Model:** CNN (Convolutional Neural Network)
   - **Task:** Generates a matrix with a random number of points (3 to 10). The model predicts the **shortest** Euclidean distance among all possible pairs of points in the matrix.

3. **`cnn_longest_distance.py`** 
   - **Model:** CNN
   - **Task:** Generates a matrix with a random number of points (3 to 10). The model predicts the **longest** Euclidean distance among all possible pairs of points in the matrix.

4. **`cnn_point_counting.py`** 
   - **Model:** CNN
   - **Task:** Generates a matrix with a random number of points (1 to 10). The model predicts the **total number of points** present in the matrix.

5. **`cnn_square_counting.py`** 
   - **Model:** CNN
   - **Task:** Generates a matrix with a random number of squares (1 to 10) of random sizes (1x1 to 5x5). The model predicts the **total number of squares** drawn in the matrix.

## Prerequisites / Requirements

To run this project, you need the following Python libraries installed:

- `numpy`
- `torch` (PyTorch)
- `matplotlib`
- `pandas` (only required in the MLP script)

You can install the required packages using pip:

```bash
pip install numpy torch torchvision matplotlib pandas
```

## How to Run

1. Open your terminal or command prompt.
2. Navigate to the project directory.
3. Run any of the scripts using Python. For example:

```bash
python mlp_distance_between_two_points.py
```

4. During execution, most scripts will prompt you to enter a **training fraction** (e.g., `0.25`, `0.50`, `1.0`). Type your choice and press Enter. This controls the proportion of the 800-sample dataset used for training.
5. After training for 20 epochs, a Matplotlib window will pop up showing the Training vs. Validation (or Test) Loss.
6. A second Matplotlib window will follow showing 10 sample predictions alongside their true target values and visualizations of the input matrices.
