### Project 1: K-Nearest Neighbors (KNN) Regressor

#### **1. Project Overview**

This project implements a K-Nearest Neighbors (KNN) Regressor from scratch, adhering to the scikit-learn API interface. The implemented model is then applied to the [Diabetes dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html) to predict disease progression. A core focus of this project is model complexity and selection, specifically utilizing internal cross-validation (CV) to determine the optimal `k` (number of neighbors) hyperparameter. The project evaluates performance using Mean Squared Error (MSE).

#### **2. Technologies Used**

* **Python:** Programming language
* **NumPy:** Numerical computing
* **Pandas:** Data manipulation and analysis
* **Matplotlib:** Data visualization
* **Scikit-learn:** Dataset loading, model evaluation utilities (for comparison/validation)

#### **3. Key Contributions & Actions**

* **Implemented KNN Regressor from Scratch:** Developed a custom KNN Regressor class (`KNNRegressor`) including `fit` and `predict` methods, mimicking the scikit-learn API.
* **Dataset Loading & Preprocessing:** Loaded and prepared the Diabetes dataset for regression tasks.
* **Model Evaluation:** Utilized Mean Squared Error (MSE) to assess the performance of the KNN Regressor.
* **Internal Cross-Validation (CV) for Hyperparameter Tuning:**
    * Designed and implemented an internal cross-validation loop to systematically search for the optimal `k` value.
    * Analyzed how the number of inner folds (`L`) affects the stability and success of internal CV.
    * Demonstrated how internal CV helps in automatic model selection, reducing the risk of manual tuning and overfitting.
* **Performance Analysis:** Conducted experiments to evaluate the model's performance on unseen data, showing the stability of `k` selection across different outer folds.

#### **4. Results & Insights**

* Successfully implemented a functional KNN Regressor from scratch, demonstrating a deep understanding of its underlying mechanics.
* The internal cross-validation procedure effectively identified near-optimal `k` values, which generalized well to outer test folds.
* Confirmed that factors such as training set size, number of inner folds, `k` value range, and data noise can affect the success of internal CV.
* This project solidified understanding of model complexity, bias-variance trade-off, and the importance of robust hyperparameter tuning techniques like cross-validation for building generalized models.

