Project Overview

This project is an end-to-end stock market analytics and prediction system that combines real-time market data, machine learning-based price prediction, and interactive Power BI dashboards to support data-driven investment insights.

The solution integrates Python (data engineering & ML) with Power BI (visual analytics), following best practices for financial data analysis and predictive modeling.

 Key Objectives

Fetch real-time and historical stock data for major Indian companies

Perform technical feature engineering (MA, volatility, RSI, returns)

Predict next-period closing prices using machine learning

Evaluate model performance using standard metrics (MAE, RMSE, MAPE)

Visualize fundamentals, predictions, and trends in Power BI

Present insights in a LinkedIn- and recruiter-ready dashboard

 Architecture Overview
Yahoo Finance / Screener.in
        ↓
Python Data Pipeline
        ↓
Feature Engineering + ML Model
        ↓
CSV Outputs
        ↓
Power BI Dashboard

 Project Structure
stock-market-prediction-powerbi/
│
├── data/
│   ├── stock_price_realtime.csv      # Real-time & historical price data
│   ├── stock_fundamentals.csv        # Fundamental metrics (PE, ROE, ROCE, Market Cap)
│   ├── stock_predictions.csv         # Actual vs Predicted prices + errors
│   └── model_metrics.csv             # MAE, RMSE, MAPE
│
├── stock_prediction_powerBI.py       # Data extraction + fundamentals pipeline
├── prediction.py                     # ML model training & prediction
├── master_pipeline.py                # Orchestrates full pipeline
├── README.md                         # Project documentation
└── dashboard.pbix                    # Power BI dashboard file

 Data Sources

Yahoo Finance (yfinance)

Real-time & historical stock prices

Market capitalization

Screener.in (Web Scraping)

P/E Ratio

ROE, ROCE

Dividend Yield

Book Value

 Feature Engineering

The following technical indicators are computed company-wise:

Daily Returns

Moving Averages (MA-5, MA-10)

Volatility (Rolling Standard Deviation)

Relative Strength Index (RSI – 14 period)

These features help capture momentum, trend, and risk behavior in stock prices.

 Machine Learning Model

Algorithm: Random Forest Regressor

Prediction Target: Next-period closing price

Training Strategy:

Company-wise models

Time-aware train-test split (no shuffling)

Model Evaluation Metrics

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

MAPE (Mean Absolute Percentage Error)

Metrics are stored in model_metrics.csv and visualized in Power BI.

 Power BI Dashboard Highlights

The dashboard is designed using a professional, pastel-themed layout, similar to industry analytics dashboards.

Key Visuals:

 KPI Cards:

Latest Market Price

Predicted Price

Expected Return (%)

Model Error (MAPE)

 Average Stock Price by Company

 Market Capitalization vs Capital Efficiency (ROE × ROCE)
 Predicted Return % by Company

 Prediction Summary Table (Latest Actual vs Predicted)

 Trading Signal

BUY / HOLD / SELL signal based on predicted return thresholds

 How to Run the Project
 Install Dependencies
pip install pandas numpy scikit-learn yfinance beautifulsoup4 requests

 Run Data & Fundamentals Pipeline
python stock_prediction_powerBI.py

Run Prediction Model
python prediction.py

 (Optional) Run Full Pipeline
python master_pipeline.py

 Open Power BI

Load CSV files from /data

Open dashboard.pbix

Refresh visuals

 Business & Analytical Insights

Combines fundamental strength (ROE, ROCE, PE) with technical momentum

Highlights stocks with high efficiency and stable predictions

Demonstrates practical ML deployment in a BI environment

 Skills Demonstrated

Python (Pandas, NumPy, Scikit-learn)

Feature Engineering & Time-Series Handling

Machine Learning (Regression)

Web Scraping

Power BI (DAX, Data Modeling, Dashboard Design)

End-to-End Data Pipeline Design

 Future Enhancements

Live API-based Power BI Service refresh

Advanced models (XGBoost, LSTM)

Sentiment analysis from financial news

Portfolio-level optimization metrics

👤 Author

Aditya Raj Sinha,
M.Sc. Data Science,
 India
