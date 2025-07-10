### Project 2: Ridge Regression with SGD & Complexity Analysis

#### **1. Project Overview**

This project delves into Ridge Regression, focusing on the derivation of the Stochastic Gradient Descent (SGD) update rule with L2 regularization. It investigates the impact of the regularization parameter (λ) on model complexity, training error, and test error, particularly in the context of polynomial regression. A key objective is to visualize and analyze the bias-variance trade-off as λ varies.

#### **2. Technologies Used**

* **Python:** Programming language
* **NumPy:** Numerical computing, especially for matrix operations and gradient calculations.
* **Matplotlib:** Data visualization for plotting error curves and model fits.
* **Scikit-learn:** Used for data generation (e.g., `make_regression`, polynomial features) and potentially as a reference for comparison.

#### **3. Key Contributions & Actions**

* **Derived SGD Update Rule for Ridge Regression:** Mathematically derived the update equations for weights in Ridge Regression when using Stochastic Gradient Descent, incorporating the L2 regularization term.
* **Implemented Ridge Regression with SGD:** Coded the Ridge Regression model with the derived SGD update rule from scratch, allowing for controlled experimentation with λ.
* **Polynomial Feature Transformation:** Applied polynomial features (up to degree 5) to the input data to increase model complexity and demonstrate the need for regularization.
* **Explored L2 Regularization Impact:** Conducted experiments by varying the regularization parameter (λ) over a wide range to observe its effects on:
    * **Model Weights:** Demonstrated how larger λ values shrink the magnitude of model coefficients.
    * **Training Error vs. Test Error:** Plotted both errors against λ on a log-log scale to visualize the bias-variance trade-off.
* **Identified Optimal Regularization:** Determined the optimal λ that minimized test Mean Squared Error (MSE), showcasing a balance between underfitting and overfitting.

#### **4. Results & Insights**

* Successfully implemented Ridge Regression with SGD, confirming the correctness of the derived update rule.
* The experiments clearly illustrated that as λ increases, model complexity decreases, leading to:
    * Reduced overfitting (lower variance) when λ is small.
    * Increased underfitting (higher bias) when λ is very large.
* The optimal λ was identified where the model generalized best, achieving the lowest test error.
* This project provided a practical understanding of how L2 regularization combats overfitting in linear models and demonstrated the critical importance of selecting an appropriate regularization strength to achieve robust generalization, especially with flexible models like high-degree polynomial regression.