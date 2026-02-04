# COVID-19 Forecasting with Facebook Prophet

## 📌 Project Overview
This project uses **Facebook Prophet**, a time-series forecasting library, to model and predict COVID-19 confirmed cases over time. 
The dataset consists of daily confirmed case counts indexed by date. The goal is to visualize trends, forecast future cases, and evaluate the performance of Prophet on epidemiological data.

## 📂 Dataset
- **Source**: COVID-19 dataset (Kaggle dataset).
- **Features Used**:
  - `date`: The reporting date of cases.
  - `confirmed`: The cumulative number of confirmed cases.

The dataset is preprocessed into Prophet’s required format:
- `ds`: Date column (datetime format).
- `y`: Target column (confirmed cases).

## ⚙️ Methodology
1. **Data Preprocessing**
   - Convert date column to `datetime`.
   - Rename columns to `ds` (date) and `y` (confirmed cases).
   - Handle missing values and ensure chronological order.

2. **Model Building**
   - Import and initialize Prophet.
   - Fit the model on the confirmed cases data.
   - Generate forecasts for a specified future period.

3. **Evaluation**
   - Compare predicted values with actual data.
   - Visualize forecast trends, confidence intervals, and seasonality components.

## 📊 Results
- Prophet produces forecasts with upper and lower confidence bounds.
- Visualization includes:
  - Historical confirmed cases.
  - Forecasted future cases.
