### Project 5: Neural Network's Decision Boundary

#### **1. Project Overview**

This project explores the ability of neural networks to learn complex, non-linear decision boundaries. It directly compares the performance of a simple Perceptron (a linear classifier) with a 3-layer feedforward neural network on a specially designed dataset that is not linearly separable. The primary goal is to visualize how the neural network transforms the input space to create an effective decision boundary, thereby achieving significantly lower classification error compared to its linear counterpart.

#### **2. Technologies Used**

* **Python:** Programming language
* **NumPy:** Numerical operations, especially for neural network computations (weights, biases, activations).
* **Pandas:** Data loading and manipulation (for the `Task2B` dataset).
* **Matplotlib:** Data visualization for plotting scatter plots and decision boundaries.
* **Scikit-learn:** Used for utility functions or potentially for generating test data.

#### **3. Key Contributions & Actions**

* **Dataset Preparation:** Loaded and visualized the `Task2B` dataset, clearly demonstrating its non-linearly separable nature.
* **Perceptron Implementation:** Implemented a Perceptron model to establish a baseline for linear separability.
    * Trained the Perceptron and calculated its classification error.
    * Visualized the linear decision boundary learned by the Perceptron.
* **3-Layer Neural Network Implementation:** Built and trained a 3-layer feedforward neural network from scratch using appropriate activation functions (e.g., sigmoid/ReLU in hidden layers and sigmoid for output) and a loss function (e.g., Mean Squared Error or Binary Cross-Entropy).
    * Utilized backpropagation for training the neural network.
    * Conducted multiple training iterations (epochs) to ensure convergence.
* **Decision Boundary Visualization:** Critically, the project visualized the decision boundary learned by the 3-layer neural network, illustrating its non-linear and complex shape.
* **Performance Comparison:** Directly compared the classification error rates of the Perceptron and the 3-layer neural network, quantifying the dramatic improvement achieved by the non-linear model.

#### **4. Results & Insights**

* The Perceptron, due to its inherent linear nature, exhibited a high classification error (e.g., 12.55%), confirming that the `Task2B` dataset is not linearly separable. Its decision boundary was a straight line, unable to effectively separate the classes.
* In stark contrast, the 3-layer neural network achieved a significantly lower error rate (e.g., 0.15%), demonstrating its capability to learn highly non-linear decision boundaries. The visualization of its decision boundary revealed a complex, curved shape that perfectly encapsulated the distinct regions of each class.
* This project unequivocally highlights the power and importance of neural networks in tackling real-world problems where underlying relationships are complex and non-linear. It solidifies the understanding that multi-layered architectures with non-linear activation functions are essential for capturing intricate patterns in data, surpassing the limitations of linear models.