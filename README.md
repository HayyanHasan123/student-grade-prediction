# student-grade-prediction
A machine learning project that uses linear regression to predict final student grades (G3) based on performance in earlier periods (G1, G2), absences, and failure history. Includes data preprocessing, exploratory data analysis, one-hot encoding, correlation heatmaps, and model evaluation with visualization

# Project Overview
In this project, we aim to use machine learning to understand how different factors (such as previous grades and attendance) affect students' final grades. We used a Linear Regression model to predict the final grades (G3) based on features like:

G1 (First Period Grades)

G2 (Second Period Grades)

Failures (Number of past class failures)

Absences (Number of absences)

# Libraries Used
pandas: Data manipulation and analysis

matplotlib & seaborn: Data visualization

sklearn: Machine learning models and evaluation

# Data Source
The data used in this project comes from the UCI Machine Learning Repository. You can access the dataset from the Student Data page.

# Key Features
Correlation Analysis: We performed correlation analysis to identify which features are most strongly related to the final grade (G3).

Linear Regression: We trained a Linear Regression model to predict students' final grades (G3) based on available features.

Visualization: Used scatter plots and a heatmap to visualize the relationships between different variables.

Model Evaluation: Evaluated the model using metrics like Mean Squared Error (MSE) and R-squared (R²) score.

# Evaluation Metrics
Mean Squared Error (MSE): This metric measures the average squared difference between the actual values (true grades) and the predicted values (predicted grades). A lower MSE indicates a better model performance because it means the predicted values are closer to the actual values.

R-squared (R²): This metric measures how well the regression predictions approximate the real data points. It provides the proportion of variance in the dependent variable (G3) that can be predicted from the independent variables. A higher R² score (closer to 1) means a better model fit, indicating that the model explains more of the variance in the final grades.

# Future Improvements
Implement more advanced machine learning models such as Random Forest or Support Vector Machines (SVM).

Use feature scaling to improve model performance.

Explore additional features such as study time, internet access, and extracurricular activities to improve the prediction accuracy.
