### Project 4: Expectation Maximization (EM) for Document Clustering

#### **1. Project Overview**

This project explores the Expectation Maximization (EM) algorithm for document clustering. It begins with the theoretical derivation of Maximum Likelihood Estimation (MLE) for a generative model of documents assigned to clusters. The core of the project involves implementing the EM algorithm to perform soft clustering, where each document is assigned a probability of belonging to each cluster. Finally, it compares the results of soft clustering (derived from EM) with a "hard" clustering approach, visualizing the differences and highlighting the benefits of probabilistic assignments.

#### **2. Technologies Used**

* **Python:** Programming language
* **NumPy:** Numerical computing, especially for probability calculations and matrix operations.
* **Pandas:** Data loading and manipulation.
* **Matplotlib:** Data visualization for scatter plots and cluster visualization.
* **Scikit-learn:** `decomposition.PCA` for dimensionality reduction to enable 2D visualization of document clusters.

#### **3. Key Contributions & Actions**

* **Derived MLE Formulations:** Mathematically derived the Maximum Likelihood Estimation (MLE) formulations for the parameters (cluster priors, word probabilities within clusters) of a generative model for document clustering. This included formalizing the EM algorithm's E-step and M-step.
* **Implemented EM Algorithm:** Developed a custom implementation of the Expectation Maximization (EM) algorithm for document clustering. This involved:
    * **E-Step (Expectation):** Calculating the posterior probability of each document belonging to each cluster (soft assignments).
    * **M-Step (Maximization):** Updating the model parameters (cluster priors and word probabilities) based on the current soft assignments.
* **Applied to Document Data:** Applied the EM algorithm to a dataset of documents (likely represented as bag-of-words vectors) to identify underlying thematic clusters.
* **Visualized Soft vs. Hard Clustering:**
    * Used Principal Component Analysis (PCA) to reduce document feature dimensions to 2D for visualization.
    * Generated two comparative scatter plots:
        * **Hard Clustering:** Each document strictly assigned to its most probable cluster.
        * **Soft Clustering:** Documents represented with varying transparency based on their assignment uncertainty (points closer to cluster boundaries are more transparent).
* **Analyzed Clustering Differences:** Compared the visual outputs to highlight:
    * Similar overall spatial cluster structures.
    * How hard clustering obscures uncertainty, while soft clustering reveals it.
    * The inherent probabilistic nature of EM outputs.

#### **4. Results & Insights**

* Successfully implemented the EM algorithm for document clustering, demonstrating its ability to learn cluster parameters and assign documents probabilistically.
* The visualization clearly showed that while both hard and soft clustering methods identify similar central cluster tendencies, **soft clustering provides crucial additional insight into the confidence and uncertainty of document assignments**.
* Documents lying at the boundaries between clusters were distinctly visible in the soft clustering plot due to their higher transparency, indicating less confident assignments.
* This project deepened the understanding of generative models, probabilistic clustering, and the iterative nature of the EM algorithm, emphasizing its advantage in providing richer, probabilistic insights over simplistic hard assignments in real-world applications like text analysis.