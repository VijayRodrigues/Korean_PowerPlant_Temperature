# ☀️ Temperature Forecast Project: Predicting Seoul's Weather with Machine Learning 🌡️

## 🚀 Overview

This project harnesses the power of machine learning to accurately predict next-day maximum and minimum air temperatures in Seoul, South Korea. We leverage a rich dataset provided by the Korea Meteorological Administration's LDAPS model, combining forecasted data with in-situ measurements and geographical features. Our goal is to build robust and reliable models that can provide valuable insights into future weather patterns.

---

## 📝 Problem Statement

Accurate temperature forecasting is crucial for various applications, including agriculture, energy management, and public safety. This project aims to enhance the precision of temperature predictions by developing machine learning models that can effectively learn from historical data and forecasted meteorological variables.

---

## 📊 Dataset

The dataset encompasses summer weather data from 2013 to 2017, specifically focusing on Seoul. It includes a variety of features such as:

* **LDAPS (Lightweight Directory Access Protocol) model forecasts**: Relative humidity, temperature, wind speed, latent heat flux, cloud cover, and precipitation.
* **In-situ temperature measurements**: Present-day maximum and minimum temperatures.
* **Geographical auxiliary variables**: Latitude, longitude, elevation, slope, and solar radiation.

---

**Download the Dataset:**

[temperature.csv](https://github.com/dsrscientist/Dataset2/blob/main/temperature.csv)

---

## 🛠️ Technologies Used

* **Python**: The backbone of our project.
* **Pandas**: For data manipulation and analysis.
* **NumPy**: For numerical computations.
* **Scikit-learn**: For machine learning model development and evaluation.
* **Matplotlib & Seaborn**: For data visualization.
* **Plotly**: For interactive geographical visualizations.
* **Statsmodels**: For statistical analysis.
* **Pandas Profiling**: For quick and comprehensive data exploration.

---

## 📁 Project Structure  

```plaintext
Temperature_Forecast_Project/
│
├── data/
│   ├── raw/
│   │   └── temperature.csv           # Original dataset
│   ├── processed/
│   │   └── cleaned_data.csv          # Processed and cleaned dataset
│   └── README.md                     # Description of the dataset
│
├── models/
│   └── Korea_Temperature_Prediction.pkl  # Saved model
│
├── notebooks/
│   ├── EDA.ipynb                     # Exploratory Data Analysis
│   ├── Model_Training.ipynb          # Model training and evaluation
│   └── Hyperparameter_Tuning.ipynb   # Hyperparameter tuning
│
├── src/
│   ├── data_preprocessing.py         # Script for data cleaning and preprocessing
│   ├── model_training.py             # Script for training and evaluating models
│   ├── utils.py                      # Utility functions (e.g., metrics, visualization)
│   └── config.py                     # Configuration file (e.g., paths, hyperparameters)
│
├── tests/
│   ├── test_data_preprocessing.py    # Unit tests for data preprocessing
│   └── test_model_training.py        # Unit tests for model training
│
├── results/
│   ├── plots/                        # Saved visualizations
│   │   ├── correlation_matrix.png
│   │   ├── feature_importance.png
│   │   └── predictions_vs_actual.png
│   └── metrics.txt                   # Model evaluation metrics
│
├── README.md                         # Project overview and instructions
├── requirements.txt                  # List of dependencies
├── LICENSE                           # License file
└── .gitignore                        # Files and directories to ignore in Git


---

## 📊 Exploratory Data Analysis (EDA)

We conducted a thorough EDA to understand the dataset's characteristics and identify potential patterns. Key findings include:

* Temporal analysis of temperature trends across years and months.
* Geographical distribution of weather stations using interactive maps.
* Correlation analysis to identify relationships between features and target variables.
* Visualizations to explore the impact of solar radiation, humidity, and cloud cover on temperature.

---

### 🗺️ Geographical Visualization

Using Plotly, we created an interactive map showcasing the distribution of weather stations in Seoul, highlighting their geographical context.

```python
import plotly.express as px

fig = px.density_mapbox(df, lat='lat', lon='lon', radius=30, opacity=0.4, height=650,
                        zoom=6.1, center=dict(lat=35.9078, lon=127.7669),
                        mapbox_style="stamen-toner")
fig.show()


---

## 🧠 Machine Learning Models

We explored various regression models to predict next-day maximum and minimum temperatures, including:

* **Linear Regression**: A fundamental model for understanding linear relationships between features and target variables.
* **Decision Tree Regressor**: A versatile model that partitions data based on feature values to make predictions.
* **Random Forest Regressor**: An ensemble method that combines multiple decision trees to improve prediction accuracy and reduce overfitting.
* **K-Nearest Neighbors Regressor**: A non-parametric model that predicts target values based on the average of the k-nearest data points.
* **AdaBoost Regressor**: An ensemble boosting technique that iteratively adjusts the weights of weak learners to build a strong predictive model.
* **ARD Regression (Automatic Relevance Determination Regression)**: A Bayesian linear regression model that automatically determines the relevance of features.


