# Census-Income-Prediction-Machine-Learning-Project

## Project Overview
This project applies Exploratory Data Analysis **(EDA)** and various **machine learning algorithms** to accurately predict individuals' income from the 1994 U.S. Census dataset. The objective is to construct a model that can predict whether an individual earns more than $50,000, a crucial piece of information for funding organizations that rely on donations. By understanding an individual's income, these organizations can tailor their requests for donations.

## Project Objectives
- Understand the data: Perform Exploratory Data Analysis (EDA) to gain insights and visualize the dataset.
- Data Cleaning: Handle missing values, inconsistencies, and outliers in the dataset.
- Preprocessing: Normalize, scale, and encode the data to prepare it for modeling.
- Model Training: Use multiple machine learning models to train the dataset, including K-Nearest Neighbors (KNN), Logistic Regression, Decision Trees (DT), and AdaBoost.
- Model Evaluation: Assess the models using accuracy, F1 score, classification reports, and confusion matrices.
- Model Optimization: Fine-tune the models to improve performance and avoid overfitting and underfitting.

## Data
The dataset consists of data collected from the 1994 U.S. Census and contains information such as age, education, hours worked per week, and income, with the goal of predicting whether an individual earns more than $50,000 a year.

## Steps
### Data Cleaning and Preprocessing
- Handle missing values.
- Encode categorical variables (e.g., using LabelEncoder).
- Scale and normalize numerical features.
- Address issues like normality, skewness, and outliers.

### Exploratory Data Analysis (EDA) and Visualization
- Visualize distributions of features.
- Identify relationships between variables using correlation plots and scatter plots.

### Modeling
- Use different algorithms to create predictive models:
  - K-Nearest Neighbors (KNN): A non-parametric method for classification.
  - Logistic Regression: A statistical model used for binary classification.
  - Decision Trees (DT): A tree-based method that splits data based on feature values.
  - AdaBoost: An ensemble technique that combines weak learners to improve accuracy.

## Model Evaluation
- Evaluate models using metrics like Accuracy, F1 Score, Confusion Matrix, and Classification Report.
- Use GridSearchCV to tune hyperparameters for better model performance.

## Results
### K-Nearest Neighbors (KNN)
- Training Accuracy: 83.59%
- Training F1 Score: 83.03%
- Testing Accuracy: 82.77%
- Testing F1 Score: 82.26%

### Logistic Regression
- Training Accuracy: 83.46%
- Training F1 Score: 82.86%
- Testing Accuracy: 84.30%
- Testing F1 Score: 83.84%

### Decision Tree (DT)
- Training Accuracy: 86.70%
- Training F1 Score: 86.30%
- Testing Accuracy: 85.10%
- Testing F1 Score: 84.90%

### AdaBoost
- F1 Score: 0.90
- Accuracy: 90%

## Model Comparison and Optimization
- AdaBoost demonstrated the best performance, with the highest F1 score and accuracy on both training and testing datasets.
- Logistic Regression also showed strong performance with stable results across both training and testing sets.
- Decision Tree and KNN had good performance, but AdaBoost outperformed them in terms of accuracy and F1 score.

## Streamlit App for Prediction Interface
To make the model more accessible, a **Streamlit** app has been developed as an interactive interface for income prediction. The app allows users to input their data and receive predictions based on the trained `.pkl` model. Additionally, the app integrates **Plotly** for creating interactive charts, providing a dynamic and engaging way to visualize the data and model predictions.

### Features of the Streamlit App:
- **User Input Form**: Users can input their demographic and socioeconomic data.
- **Prediction Output**: The app predicts whether the user's income is likely to be above or below $50,000.
- **Interactive Visualizations**: Plotly is used to create interactive charts that help users understand the data and model predictions better.

### How to Run the Streamlit App:
1. Clone the repository.
2. Run the Streamlit app using the command `streamlit run app.py`.
3. Open the provided URL in your web browser to access the app.

## Conclusion
- In this project, we successfully applied various machine learning algorithms to predict whether an individual earns more than $50,000 based on census data. The AdaBoost model provided the best results overall, but all models showed strong performance. The use of proper preprocessing and understanding the assumptions of each model allowed us to optimize performance and avoid common pitfalls like overfitting and underfitting.
- The integration of a Streamlit app with Plotly visualizations provides an interactive and user-friendly way to explore the model's predictions, making it easier for non-technical users to understand and utilize the results.
