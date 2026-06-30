# traffic-flow-forecasting-xgboost 
<h1 align="center">🚦 Multi-Horizon Traffic Flow Forecasting using XGBoost</h1>

<p align="center">
  A machine learning time-series forecasting project to predict short-term traffic flow for the next 15, 30, 45, and 60 minutes.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge">
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">
  <img src="https://img.shields.io/badge/Time_Series-Forecasting-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Jupyter_Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white">
</p>

---

## 📌 Project Overview

This project uses machine learning to forecast short-term traffic flow using historical traffic sensor data.

The model predicts traffic flow for multiple future horizons:

* **+15 minutes**
* **+30 minutes**
* **+45 minutes**
* **+60 minutes**

The workflow includes data cleaning, time conversion, feature engineering, lag feature creation, exploratory data analysis, scaling, chronological train-validation-test splitting, XGBoost modelling, hyperparameter tuning, and forecast evaluation.

The final model uses **XGBoost Regressor** trained separately for each forecast horizon.

---

## 🎯 Business Problem

Traffic flow forecasting is important for smart city planning, transport management, congestion monitoring, and real-time traffic decision-making.

Accurate short-term forecasting can help:

* Predict future traffic congestion
* Support road network monitoring
* Improve transport planning
* Assist traffic control systems
* Help commuters and authorities make better decisions

This project demonstrates how machine learning can be used to predict near-future traffic flow from historical sensor data.

---

## 📊 Dataset Information

The dataset used is:

```text
traffic_flow.csv
```

The dataset contains:

```text
1,048,575 rows
12 original columns
```

### Original Columns

| Column     | Description                          |
| ---------- | ------------------------------------ |
| site       | Traffic monitoring site or sensor ID |
| day        | Day abbreviation                     |
| date       | Date of traffic record               |
| start_time | Start time of 15-minute interval     |
| end_time   | End time of 15-minute interval       |
| flow       | Traffic flow count                   |
| flow_pc    | Traffic flow percentage              |
| cong       | Congestion value                     |
| cong_pc    | Congestion percentage                |
| dsat       | Degree of saturation                 |
| dsat_pc    | Degree of saturation percentage      |
| ObjectId   | Unique object identifier             |

---

## 🛠️ Tools and Libraries Used

| Tool / Library   | Purpose                                         |
| ---------------- | ----------------------------------------------- |
| Python           | Data analysis and machine learning              |
| Pandas           | Data loading, cleaning, and feature engineering |
| NumPy            | Numerical operations                            |
| Matplotlib       | Data visualization                              |
| Seaborn          | Statistical visualization                       |
| Scikit-learn     | Scaling, splitting, and evaluation              |
| XGBoost          | Gradient boosting regression model              |
| Statsmodels      | Seasonal decomposition                          |
| Jupyter Notebook | Interactive development environment             |

---

## ✅ Project Workflow

* ✅ Imported required Python libraries
* ✅ Loaded traffic flow dataset
* ✅ Checked data types and dataset structure
* ✅ Removed duplicate records
* ✅ Converted day abbreviations into full day names
* ✅ Converted start and end time columns into datetime format
* ✅ Extracted hour and day-of-week features
* ✅ Checked and handled missing values
* ✅ Created exploratory visualizations
* ✅ Created lag features from previous traffic flow values
* ✅ Created daily seasonal lag features
* ✅ Created multi-horizon target variables
* ✅ Scaled input and output variables using MinMaxScaler
* ✅ Split data chronologically into train, validation, and test sets
* ✅ Built baseline MultiOutput XGBoost model
* ✅ Evaluated baseline model using MSE, MAE, and R²
* ✅ Tuned XGBoost hyperparameters
* ✅ Trained final horizon-wise XGBoost models
* ✅ Evaluated forecasting performance for 15, 30, 45, and 60-minute horizons
* ✅ Visualized actual vs predicted traffic flow

---

## 📈 Exploratory Data Analysis

The notebook includes:

* 📈 7-day rolling average traffic flow trend
* 📊 Average traffic flow by month
* 🔥 Average traffic flow by hour and day heatmap
* 📉 One-day-before traffic flow distribution
* 📊 Seasonal decomposition of traffic flow
* 📈 Actual vs predicted forecast line plots

---

## 🧠 Feature Engineering

The project creates lag-based time-series features.

### Short-Term Lag Features

| Feature | Meaning                        |
| ------- | ------------------------------ |
| lag_1   | Traffic flow 15 minutes before |
| lag_2   | Traffic flow 30 minutes before |
| lag_3   | Traffic flow 45 minutes before |
| lag_4   | Traffic flow 60 minutes before |

### Daily Seasonal Lag Features

| Feature   | Meaning                                  |
| --------- | ---------------------------------------- |
| flow_t-94 | Traffic flow around one day before       |
| flow_t-95 | Traffic flow around one day before       |
| flow_t-96 | Traffic flow exactly 96 intervals before |

Since the dataset uses 15-minute intervals, **96 intervals = 24 hours**.

### Time-Based Features

| Feature   | Meaning                 |
| --------- | ----------------------- |
| hour      | Hour of the day         |
| dayofweek | Day of week as a number |

---

## 🎯 Target Variables

The model predicts four future traffic flow values:

| Target   | Forecast Horizon |
| -------- | ---------------- |
| flow_t+1 | Next 15 minutes  |
| flow_t+2 | Next 30 minutes  |
| flow_t+3 | Next 45 minutes  |
| flow_t+4 | Next 60 minutes  |

---

## 🤖 Model Used

The main model used is:

```python
XGBRegressor(objective="reg:squarederror")
```

Two modelling approaches were used:

1. **Baseline MultiOutputRegressor with XGBoost**
2. **Final horizon-wise XGBoost models after hyperparameter tuning**

---

## 📊 Baseline Model Performance

The baseline XGBoost model achieved:

| Metric   | Score    |
| -------- | -------- |
| MSE      | 13247.42 |
| MAE      | 52.34    |
| R² Score | 0.7116   |

This shows that the baseline model was able to capture a meaningful traffic flow pattern, but further tuning was required to improve horizon-wise performance.

---

## ⚙️ Hyperparameter Tuning

The notebook tested different XGBoost parameter combinations using validation performance.

Tuned parameters included:

* Learning rate
* Maximum tree depth
* Subsample ratio
* Column sample ratio
* Number of estimators
* Minimum child weight
* Regularization values

The final model was trained separately for each forecast horizon.

---

## 📊 Final Model Performance

The final tuned XGBoost models achieved the following performance:

| Forecast Horizon | MAE    | R² Score |
| ---------------- | ------ | -------- |
| +15 minutes      | 0.0273 | 0.7936   |
| +30 minutes      | 0.0285 | 0.7835   |
| +45 minutes      | 0.0298 | 0.7704   |
| +60 minutes      | 0.0319 | 0.7495   |

---

## 🔍 Key Observations

* The model performs best for the **+15 minute forecast horizon**.
* Forecasting accuracy decreases as the prediction horizon increases.
* The **+60 minute forecast** is harder because traffic patterns become less certain further into the future.
* Lag features are important because recent traffic flow strongly influences near-future traffic flow.
* Daily seasonal lag features help the model understand repeated traffic patterns.
* XGBoost is effective for structured time-series forecasting when lag and time-based features are engineered properly.

---

## 📉 Forecast Horizon Analysis

The results show a clear horizon deterioration pattern:

| Horizon | Interpretation                                                            |
| ------- | ------------------------------------------------------------------------- |
| +15 min | Best accuracy because the target is closest to current traffic conditions |
| +30 min | Slight performance decrease                                               |
| +45 min | More uncertainty appears                                                  |
| +60 min | Lowest R² due to longer forecast distance                                 |

This is expected in time-series forecasting because future values become harder to predict as the time horizon increases.

---

## 💼 Business Use Case

This project can be used as a smart mobility and transport analytics prototype for:

* 🚦 Short-term traffic forecasting
* 🏙️ Smart city traffic monitoring
* 🚗 Congestion prediction
* 📊 Road network performance analysis
* 🚌 Public transport planning
* 🧠 Intelligent transport systems
* 📈 Time-series machine learning portfolio demonstration

The project shows how traffic sensor data can be transformed into predictive features and used for short-term forecasting.

---

## 📁 Folder Structure

```text
traffic-flow-forecasting-xgboost/
│
├── README.md
├── notebook/
│   └── traffic_flow_forecasting_xgboost.ipynb
│
├── data/
│   └── traffic_flow.csv
│
└── docs/
    └── model-insights.md
```

---

## 📂 Project Files

The main notebook is available in the `notebook/` folder:

```text
notebook/traffic_flow_forecasting_xgboost.ipynb
```

The dataset should be stored in the `data/` folder:

```text
data/traffic_flow.csv
```

---

## 🎯 What I Learned

* ✅ Loading and cleaning large traffic sensor datasets
* ✅ Converting date and time columns into datetime format
* ✅ Creating time-based features such as hour and day of week
* ✅ Creating lag features for time-series forecasting
* ✅ Creating multi-horizon prediction targets
* ✅ Using rolling trend analysis and seasonal decomposition
* ✅ Building XGBoost regression models
* ✅ Using MultiOutputRegressor for multi-target forecasting
* ✅ Performing chronological train-validation-test splitting
* ✅ Evaluating regression models using MAE, MSE, RMSE, and R²
* ✅ Performing horizon-wise model evaluation
* ✅ Understanding forecast horizon deterioration
* ✅ Preparing a time-series machine learning project for GitHub portfolio

---

## 🔮 Future Improvements

* 🔲 Add site-wise traffic forecasting
* 🔲 Add rolling mean and rolling standard deviation features
* 🔲 Add weather or event data as external features
* 🔲 Add feature importance analysis
* 🔲 Add SHAP explainability for XGBoost
* 🔲 Compare XGBoost with Random Forest, Linear Regression, LSTM, GRU, and Prophet
* 🔲 Add RMSE table for each forecast horizon
* 🔲 Improve test-set evaluation naming and avoid validation/test confusion
* 🔲 Build a Streamlit dashboard for traffic forecasting
* 🔲 Deploy the forecasting model as a traffic prediction demo

---

## ⚠️ Important Note

This project is created for educational and portfolio purposes.

The notebook currently demonstrates a strong forecasting workflow, but before using it in production, the model should be validated with a clean test set, site-wise split checks, and real-time data assumptions.

---

## 👨‍💻 Author

**Saran Kumar Krishnan**

<p>
  <img src="https://img.shields.io/badge/GitHub-SaranStiff02-black?style=for-the-badge&logo=github">
  <img src="https://img.shields.io/badge/Project-Time_Series_Forecasting-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge"> </p
