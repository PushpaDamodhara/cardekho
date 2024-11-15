Car Dekho - Used Car Price Prediction
Project Overview
This project aims to predict the prices of used cars listed on CarDekho using machine learning models. The goal is to create an interactive and user-friendly Streamlit web application that helps customers and sales representatives estimate car prices based on various features such as car make, model, year, fuel type, and more.

Skills & Tools
Data Cleaning and Preprocessing
Exploratory Data Analysis (EDA)
Machine Learning Model Development
Price Prediction Techniques
Model Evaluation and Optimization
Streamlit Application Development
Documentation and Reporting
Domain
Automotive Industry
Data Science
Machine Learning
Libraries Used
Pandas: For data manipulation and analysis.
NumPy: For numerical operations and handling arrays.
Matplotlib: For data visualization.
Seaborn: For statistical data visualization.
Scikit-learn: For machine learning algorithms, model evaluation, and preprocessing.
Pickle: For saving and loading Python objects
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pickle

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

Problem Statement
The goal of the project is to improve the customer experience by building a machine learning model that accurately predicts the prices of used cars. The model will be integrated into a web-based application using Streamlit, allowing users to input car details and receive price predictions in real time.

Objective
Develop a machine learning model to predict used car prices based on various car features.
Integrate the model into a Streamlit application to provide real-time price predictions.
Data Source
The dataset was collected from CarDekho, featuring detailed information on used cars from multiple cities. Each city's data was stored in separate Excel files, which include car features, specifications, and availability details.

Project Scope
Data Processing:

Import and Concatenate: Import datasets from different cities, convert unstructured data into structured format, and concatenate them into a single dataset with an additional 'City' column.
Handling Missing Values: Use imputation techniques such as mean, median, or mode for numerical columns, and mode or a new category for categorical columns.
Standardizing Data Formats: Ensure consistency by removing units (e.g., 'kms' from numerical columns) and converting all relevant fields to the correct data types.
Encoding Categorical Variables:

Apply label encoding for ordinal categories (e.g., fuel types, transmission types).
Exploratory Data Analysis (EDA)
Perform descriptive statistics (mean, median, mode, etc.) to understand the distribution of the data.
Use visualizations such as scatter plots, histograms, box plots, and correlation heatmaps to identify patterns and relationships.
Feature selection using correlation analysis and feature importance metrics.
Model Development
Train-Test Split:

Split the dataset into training and testing sets (70-30 or 80-20 split).
Model Selection:

Evaluate various models, including Linear Regression, Decision Trees, Random Forests, and Gradient Boosting.
Model Training:

Train the selected models and use cross-validation to ensure robust performance.
Model Evaluation
Use metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²) to evaluate model performance.
Compare the performance of different models and select the best-performing one.
Deployment
Deploy the model using Streamlit to create an intuitive web application.
Enable users to input car features and obtain real-time price predictions.
Model Performance Summary
Gradient Boosting: Best performer with an R-squared value of 0.94, demonstrating high accuracy.
Random Forest: Strong performance with an R-squared value of 0.93.
Decision Tree: Moderate performance with an R-squared value of 0.78.
Linear Regression: Weakest model with an R-squared value of 0.59.
