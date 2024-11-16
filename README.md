# Predictive Analysis on Credit Score Dataset

This project applies various machine learning techniques to a credit score dataset to predict loan risk levels for customers. By exploring both supervised and unsupervised learning models, the project demonstrates and compares the effectiveness of multiple algorithms, providing a hands-on learning experience.

---

## Project Overview

The primary goal of this project was to build, train, and test predictive models on a Kaggle credit score dataset. The dataset includes customer attributes like ID, gender, income, and education as predictor variables. Since a target variable was not provided, I created a new column `STATUS` with random binary values (0 and 1), representing "low risk" and "high risk" categories. While random assignment does not yield precise real-world predictions, it enables practical model testing and comparisons.

## Steps in the Project

1. **Data Preprocessing and Cleaning**
   - Loaded the dataset and removed columns with minimal impact on prediction, like `ID`, `CODE_GENDER`, and `FLAG_EMAIL`.
   - Converted `DAYS_BIRTH` and `DAYS_EMPLOYED` to years and took absolute values to ensure positivity.
   - Transformed categorical columns (e.g., `FLAG_OWN_CAR`, `NAME_INCOME_TYPE`) into factors for compatibility.

2. **Data Normalization and Encoding**
   - Applied normalization to numeric columns for consistent scaling.
   - Encoded categorical columns numerically, converting factor levels to integers to work with all models.

3. **Train-Test Split**
   - Split data into 70% training and 30% testing sets. This partition provides an unbiased accuracy estimate by evaluating models on unseen data.

4. **Modeling**
   - **K-Nearest Neighbors (KNN)**: Used KNN with `k=10` as a baseline model to predict loan risk, assessing accuracy by comparing predictions against actual labels.
   - **Naive Bayes**: Implemented Naive Bayes (using the `e1071` library), a probabilistic approach that assumes independence among predictors.
   - **Decision Tree**: Trained a decision tree classifier to predict loan risk, visualized with `rpart.plot` to observe decision paths.
   - **K-Means Clustering (Unsupervised)**: Applied K-means clustering with two clusters to explore patterns in the data, visualizing cluster centers and groupings.

5. **Accuracy Comparisons**
   - Calculated and compared accuracies for KNN, Naive Bayes, and Decision Tree models.
   - Created a bar chart with `ggplot2` to compare model accuracies visually.

## Visualizations
- **Decision Tree**: Visualized to display classification rules.
- **K-Means Clustering Plot**: Displays data points grouped by clusters, showing natural groupings.
- **Accuracy Comparison Chart**: Highlights accuracy for each model to identify the best-performing algorithm.

## Libraries and Tools Used
- **Data Manipulation**: `dplyr`, `lapply`, `as.factor`
- **Modeling**: `class` (KNN), `e1071` (Naive Bayes), `rpart` (Decision Tree), `kmeans`
- **Visualization**: `ggplot2` (bar chart), `rpart.plot` (decision tree visualization)

## Conclusion
This project demonstrates a comprehensive approach to predictive analysis using multiple machine learning models. Although the dataset lacks a genuine target variable, the random target column enables an educational exploration of various classification techniques.
