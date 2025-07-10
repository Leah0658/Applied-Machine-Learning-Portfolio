### Project 3: Classifier Comparison - Logistic Regression vs. Bayes Classifiers

#### **1. Project Overview**

This project provides a comprehensive comparison of several classification algorithms: Logistic Regression, Naive Bayes, Gaussian Bayes Classifier (with shared covariance), and Gaussian Bayes Classifier (with full covariance). The objective is to analyze their performance on the Breast Cancer Wisconsin (Diagnostic) dataset, particularly examining how their accuracy varies with different training data sizes. The project highlights the bias-variance trade-off in the context of model complexity and data availability.

#### **2. Technologies Used**

* **Python:** Programming language
* **NumPy:** Numerical operations
* **Pandas:** Data handling and manipulation
* **Matplotlib:** Data visualization for plotting accuracy curves
* **Scikit-learn:**
    * `datasets.load_breast_cancer`: Dataset loading
    * `LogisticRegression`: Logistic Regression implementation
    * `GaussianNB`: Naive Bayes Classifier
    * `GaussianMixture`: Used to model Gaussian components for Bayes Classifiers
    * `train_test_split`: Data splitting
    * `accuracy_score`: Performance evaluation

#### **3. Key Contributions & Actions**

* **Implemented Multiple Classifiers:** Set up and configured four distinct classification models:
    * Logistic Regression
    * Naive Bayes Classifier
    * Gaussian Bayes Classifier with shared covariance (Linear Discriminant Analysis - LDA equivalent structure)
    * Gaussian Bayes Classifier with full covariance (Quadratic Discriminant Analysis - QDA equivalent structure)
* **Dataset Preparation:** Loaded and preprocessed the Breast Cancer Wisconsin dataset for classification.
* **Performance Evaluation Across Training Sizes:**
    * Developed a framework to train and evaluate all four models repeatedly on varying proportions of the training data (e.g., 2%, 4%, ..., 100%).
    * Plotted training and test accuracy for each model as a function of the training set size.
* **Bias-Variance Trade-off Analysis:** Analyzed how each model's performance changes with data size, specifically observing:
    * **Logistic Regression:** Demonstrated strong performance and robustness even with limited data, suggesting a good balance of bias and variance.
    * **Naive Bayes:** Showcased its simplicity and effectiveness, particularly in scenarios with limited data due to its strong independence assumptions.
    * **Gaussian Bayes (Shared Covariance):** Performed well, representing a more flexible model than Naive Bayes but less complex than full covariance, indicating a reasonable bias-variance trade-off.
    * **Gaussian Bayes (Full Covariance):** Showed high training accuracy but significant drops in test accuracy with small datasets, highlighting its susceptibility to overfitting when data is scarce due to its higher parameter count.
* **Model Comparison:** Provided insights into the strengths and weaknesses of each classifier based on dataset characteristics and training data availability.

#### **4. Results & Insights**

* The experiments clearly demonstrated that models with fewer parameters (Logistic Regression, Naive Bayes) tend to perform better with limited data, exhibiting higher bias but lower variance.
* More complex models (Gaussian Bayes with full covariance) require more data to accurately estimate parameters and generalize well. With insufficient data, they are prone to overfitting, leading to high variance.
* The project reinforced the understanding of the bias-variance trade-off, showing how different model complexities interact with data availability to impact generalization performance.
* For the given breast cancer dataset, Logistic Regression often proved to be a robust performer, even with limited training examples, suggesting the underlying decision boundary is relatively simple.
