# Healthcare Data Science Project – Bhutan

A Python-based data science workflow for analyzing healthcare data relevant to Bhutan, supported by a Streamlit application for interactive visualization and model inference.

This project is designed as a template that guides users through data acquisition, preprocessing, exploratory data analysis (EDA), feature engineering, machine learning modeling, and deployment of a Streamlit application.

## 1. Project Objectives

Analyze healthcare data focused on Bhutan (e.g., disease prevalence, hospital admissions, health indicators, demographic trends).

Develop an end-to-end data science workflow in Python.

Build and deploy an interactive Streamlit dashboard showcasing insights and model outputs.

Demonstrate reproducible pipelines for healthcare analytics and prediction tasks.

## 2. Project Structure

healthcare-bhutan/
│
├── data/
│ ├── raw/ # Raw datasets (CSV, XLSX, etc.)
│ ├── processed/ # Cleaned & transformed data
│
├── notebooks/
│ ├── 01_data_exploration.ipynb
│ ├── 02_feature_engineering.ipynb
│ ├── 03_modeling.ipynb
│
├── src/
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── train_model.py
│ ├── utils.py
│
├── app/
│ ├── streamlit_app.py
│
├── models/
│ ├── trained_model.pkl
│
├── README.md
├── requirements.txt
└── starter.py

## 3. Data Requirements

You may obtain healthcare data relevant to Bhutan from sources such as:

Bhutan Ministry of Health datasets

Bhutan e-Government Open Data Portal

WHO Global Health Observatory

World Bank Health Indicators

Kaggle public datasets related to South Asian health metrics

All files should be placed in data/raw/.
Download data set from https://data.humdata.org/dataset/who-data-for-btn

## 4. Environment Setup

### 4.1 Python Version

Use Python 3.10+.

### 4.2 Create Virtual Environment

python -m venv venv
source venv/bin/activate # macOS/Linux
venv\Scripts\activate # Windows

### 4.3 Install Requirements

pip install -r requirements.txt

### 4.4 Example requirements.txt

Copy into your project:

pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
joblib
pyyaml

## 5. Workflow Instructions

### 5.1 Step 1 – Data Loading

Use starter.py or src/data_loader.py to load raw healthcare data:

Validate CSV structure

Check for missing values

Ensure correct data types

Store cleaned outputs into /data/processed

### 5.2 Step 2 – Preprocessing

Implement:

Missing value handling

Feature scaling (if needed)

Outlier detection

Encoding of categorical variables

Time-series alignment (if applicable)

Save transformed data.

### 5.3 Step 3 – Exploratory Data Analysis (EDA)

Perform analyses in Jupyter notebooks:

Disease prevalence by Dzongkhag

Time trend of cases

Hospital admission patterns

Correlation matrices

Demographic distribution analysis

Export charts for the Streamlit app.

### 5.4 Step 4 – Feature Engineering

Based on EDA:

Construct aggregated health indicators

Build risk scores

Generate lagged variables for time-series forecasting

Normalization or binning of numerical variables

### 5.5 Step 5 – Machine Learning Modeling

Choose a modeling approach:

Classification: disease risk prediction

Regression: patient load forecasts

Clustering: district-wise segmentation

Steps:

Split data into training/testing sets.

Train candidate models.

Evaluate using metrics (Accuracy, F1, RMSE).

Save final model as .pkl in /models.

## 6. Streamlit App Deployment

### 6.1 Local Deployment

Run:

streamlit run app/streamlit_app.py

### 6.2 What the Streamlit App Includes

Load processed healthcare data

Display trend charts

Model prediction widget

Key health indicators dashboard

District-wise comparisons for Bhutan

## 7. Deployment on Streamlit Cloud

Push repository to GitHub

Go to https://streamlit.io/cloud

Connect GitHub repository

Select file: app/streamlit_app.py

Add dependencies in requirements.txt

Deploy

Updates to GitHub redeploy automatically.

## 8. Future Enhancements

Add geospatial visualizations (Bhutan district maps)

Integrate live data feeds

Create a REST API endpoint for predictions

Add user authentication

Expand to multivariate forecasting models

## 9. License

Specify your chosen license (MIT recommended).
