# DA5401 A4: GMM-Based Synthetic Sampling for Imbalanced Data

## 1. Project Overview

This project implements and evaluates a **Gaussian Mixture Model (GMM)** as an advanced technique for handling severe class imbalance in the credit card fraud detection domain. The primary goal is to generate high-quality synthetic data for the minority (fraud) class to create a balanced training set.

The performance of a **Logistic Regression** classifier trained on the GMM-balanced data is compared against a baseline model trained on the original, imbalanced data to assess the effectiveness of this probabilistic sampling approach.

## 2. Dataset

The project uses the well-known **Credit Card Fraud Detection** dataset from Kaggle, loaded from `creditcard.csv`.

- **Shape**: The dataset contains 284,807 transactions and 31 columns.
- **Features**: Features include `Time`, `Amount`, and 28 anonymized principal components (`V1` to `V28`).
- **Class Imbalance**: The dataset is highly imbalanced, with fraudulent transactions making up only **0.17%** of the total data, resulting in an imbalance ratio of approximately **578:1** (Normal:Fraud).

## 3. Methodology & Key Steps

The notebook is structured in three main parts: Baseline Model Training, GMM-based Resampling, and Comparative Evaluation.

### Part A: Baseline Model and Data Analysis
1.  **Data Preprocessing**: The `Time` and `Amount` features are scaled using `StandardScaler` to normalize their ranges.
2.  **Data Splitting**: The data is split into an 80% training set and a 20% test set, using stratification to preserve the original class distribution.
3.  **Baseline Model**: A Logistic Regression classifier is trained on the original, imbalanced, and scaled training data. This model serves as the performance benchmark.

### Part B: Gaussian Mixture Model (GMM) for Synthetic Sampling
1.  **Theoretical Foundation**: GMM is chosen over simpler methods like SMOTE because of its ability to model complex, multi-modal probability distributions, which can better represent diverse fraud patterns.
2.  **Optimal Component Selection**: A GMM is fitted on the minority (fraud) class samples from the training data. The optimal number of Gaussian components is determined by comparing AIC and BIC scores, with the more conservative **BIC** suggesting an optimal number of **3 components**.
3.  **Synthetic Data Generation**: Two resampling strategies are implemented:
    -   **GMM Full Balance**: The GMM is used to generate enough synthetic minority samples to achieve a 1:1 balance with the original majority class.
    -   **GMM + CBU (Clustering-Based Undersampling)**: The majority class is first undersampled using K-Means clustering, and then the GMM generates synthetic minority samples to match the size of the reduced majority class, creating a smaller, balanced dataset.

### Part C: Performance Evaluation
1.  **Model Training**: Two new Logistic Regression models are trained on the `GMM Full Balance` and `GMM + CBU` datasets.
2.  **Comparative Analysis**: All three models (Baseline, GMM Full, GMM + CBU) are evaluated on the original, untouched imbalanced test set. The key metrics—Precision, Recall, and F1-Score—are compiled and visualized for comparison.

## 4. Results

The evaluation showed that while GMM-based resampling dramatically increased recall, it came at the cost of a severe drop in precision, leading to a poorer overall F1-Score compared to the baseline.

| Model              | Precision | Recall | F1-Score |
| :----------------- | :-------- | :----- | :------- |
| **Baseline** | **0.8267** | 0.6327 | **0.7168** |
| GMM Full Balance   | 0.0826    | **0.8980** | 0.1512   |
| GMM + CBU          | 0.0789    | 0.8878 | 0.1449   |

-   **Baseline Model**: Achieved an F1-Score of **0.7168**, with a high precision of 0.8267 but a moderate recall of 0.6327. This means when it flags a transaction, it is correct ~83% of the time, but it only catches ~63% of all actual frauds.
-   **GMM Models**: Both GMM-based models successfully boosted recall to **~90%**, significantly improving the detection of fraudulent transactions. However, this was accompanied by a catastrophic precision drop to **~8%**, resulting in a very high number of false positives and a poor F1-Score (~0.15).

## 5. Conclusion

The notebook's final recommendation is to **adopt GMM-based synthetic sampling**, citing its theoretical strengths and the significant improvement in recall.

However, the empirical results from the experiment show a different story. The **F1-Score for the GMM-based models dropped by ~80%** compared to the baseline. The extremely low precision of the GMM models (~0.08) would make them impractical for real-world deployment, as they would generate over 10 false alarms for every one correct fraud detection.

Therefore, based on the quantitative results, the **Baseline model provides the most balanced and operationally viable performance** with the highest F1-Score. While GMM is a powerful technique, this analysis highlights that it requires careful tuning to avoid degrading precision to unusable levels.

## 6. How to Run

1.  **Prerequisites**: Ensure you have Python 3 and the following libraries installed:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

2.  **Dataset**: Download the `creditcard.csv` file and place it in the same directory as the notebook.

3.  **Execution**: Open the `.ipynb` file in a Jupyter environment (like Jupyter Notebook, JupyterLab, or Google Colab) and run the cells sequentially from top to bottom.
