import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from collections import Counter
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the data
customer_demographic = pd.read_csv('data/CustomerDemographics.csv')
productsinfo = pd.read_csv('data/ProductInfo.csv')
transactional_data_01 = pd.read_csv('data/Transactional_data_retail_01.csv')
transactional_data_02 = pd.read_csv('data/Transactional_data_retail_02.csv')

# Merge transactional data
transactional_data = pd.concat([transactional_data_01, transactional_data_02])

# Merge transactional data with product info
transactional_data = transactional_data.merge(productsinfo, on='StockCode', how='inner')

# Merge transactional data with customer demographic info
# transactional_data = transactional_data.merge(customer_demographic, on='Customer ID', how='inner')

# transactional_data having negative quantity and price

# Make both Quantity and Price positive
transactional_data['Quantity'] = np.abs(transactional_data['Quantity'])
transactional_data['Price'] = np.abs(transactional_data['Price'])

# print((transactional_data['Price'] < 0).sum())
# print((transactional_data['Quantity'] < 0).sum())

# prompt: create a new column called 'Revenue' which is the product of Quantity and Price
transactional_data['Revenue'] = transactional_data['Quantity'] * transactional_data['Price']

# drop invoice, Description column
transactional_data.drop(['Invoice'], axis=1, inplace=True)

# sort the transactional data by 'Revenue' in ascending order
transactional_data.sort_values(by='Revenue', ascending=False, inplace=True)

# Prepare the data: Combine customer demographic data with transactional data
# Merge transactional data with customer demographic info
customer_demographic_productinfo = pd.merge(customer_demographic, transactional_data, on='Customer ID', how='inner')

# Time Series Techniques: Apply time series models (e.g., ARIMA, Prophet) to predict demand based on historical data.
# Ensure the data is sorted by time
# # problem faced with the InvoiceDate column with the format '%d %B %Y', first put '%Y-%m-%d'
# Define the list of date formats
date_formats = [
    '%d %B %Y',  # e.g., 15 October 2022
    '%Y-%m-%d',  # e.g., 2022-10-15
    '%d/%m/%Y',  # e.g., 15/10/2022
    '%m/%d/%Y',  # e.g., 10/15/2022
    '%d-%m-%Y',  # e.g., 15-10-2022
    '%m-%d-%Y'   # e.g., 10-15-2022
]

# Function to parse dates with multiple formats
def parse_date(date_str):
    for date_format in date_formats:
        try:
            return pd.to_datetime(date_str, format=date_format)
        except ValueError:
            continue
    return pd.NaT

# Apply the function to the InvoiceDate column
customer_demographic_productinfo['InvoiceDate'] = customer_demographic_productinfo['InvoiceDate'].apply(parse_date)

# NaT values are dropped from the data, found in the InvoiceDate column
customer_demographic_productinfo = customer_demographic_productinfo.dropna(subset=['InvoiceDate'])

customer_demographic_productinfo = customer_demographic_productinfo.sort_values(by='InvoiceDate')

# Ensure the InvoiceDate column is parsed as datetime and set as the index
customer_demographic_productinfo['InvoiceDate'] = pd.to_datetime(customer_demographic_productinfo['InvoiceDate'])

# Define a custom aggregation function for StockCode to get the most common value
def most_common(values):
    if len(values) == 0:
        return None
    return Counter(values).most_common(1)[0][0]

# Set the index to InvoiceDate
customer_demographic_productinfo.set_index('InvoiceDate', inplace=True)

# Resample the data on a weekly basis and aggregate
weekly_data = customer_demographic_productinfo.resample('W').agg({
    'StockCode': most_common,  # Apply custom aggregation for StockCode
    'Revenue': 'sum',          # Sum for numeric columns
    'Quantity': 'sum'          # Sum for numeric columns
    # Add other columns and their aggregation methods as needed
})

# Reset the index to make InvoiceDate a column again
weekly_data.reset_index(inplace=True)

# Filter for the top 10 products by revenue
top_10_products = customer_demographic_productinfo.groupby('StockCode')['Quantity'].sum().nlargest(10).index
weekly_top_10 = weekly_data[weekly_data['StockCode'].isin(top_10_products)]

# Ensure the index is a datetime index
weekly_top_10.set_index('InvoiceDate', inplace=True)

# Drop any missing values
weekly_top_10 = weekly_top_10.dropna()

# Retain the original StockCode column for filtering
weekly_top_10_with_stockcode = weekly_top_10.copy()

# Apply OneHotEncoder to categorical data
categorical_columns = ['StockCode']  # Add other categorical columns if needed
encoder = OneHotEncoder(sparse_output=False)
encoded_categorical_data = encoder.fit_transform(weekly_top_10[categorical_columns])
encoded_categorical_df = pd.DataFrame(encoded_categorical_data, columns=encoder.get_feature_names_out(categorical_columns))



# Drop original categorical columns and concatenate encoded columns
weekly_top_10 = weekly_top_10.drop(categorical_columns, axis=1)
weekly_top_10 = pd.concat([weekly_top_10.reset_index(drop=True), encoded_categorical_df], axis=1)

# Define the time-based cross-validation strategy
tscv = TimeSeriesSplit(n_splits=5)

# Initialize lists to store scores and predictions
dt_rmse_scores = []
dt_mae_scores = []
xgb_rmse_scores = []
xgb_mae_scores = []
dt_predictions_all = []
xgb_predictions_all = []
actual_all = []

# Train and evaluate the models using time-based cross-validation
for train_index, test_index in tscv.split(weekly_top_10):
    X_train, X_test = weekly_top_10.iloc[train_index], weekly_top_10.iloc[test_index]
    y_train, y_test = weekly_top_10['Revenue'].iloc[train_index], weekly_top_10['Revenue'].iloc[test_index]
    
    # Train a DecisionTree model
    dt_model = DecisionTreeRegressor()
    dt_model.fit(X_train, y_train)
    dt_predictions = dt_model.predict(X_test)
    dt_rmse_scores.append(root_mean_squared_error(y_test, dt_predictions))
    dt_mae_scores.append(mean_absolute_error(y_test, dt_predictions))
    dt_predictions_all.extend(dt_predictions)
    
    # Train an XGBoost model
    xgb_model = XGBRegressor()
    xgb_model.fit(X_train, y_train)
    xgb_predictions = xgb_model.predict(X_test)
    xgb_rmse_scores.append(root_mean_squared_error(y_test, xgb_predictions))
    xgb_mae_scores.append(mean_absolute_error(y_test, xgb_predictions))
    xgb_predictions_all.extend(xgb_predictions)

    # Store the actual values
    actual_all.extend(y_test)

# Streamlit app
st.title("Top 10 Products Analysis")

# Create a two-column layout
col1, col2 = st.columns([1, 3])

# Left column for "Top 10 Products by Quantity"
with col1:
    st.write("Top 10 Products by Quantity:")
    st.write(top_10_products)

# Right column for "Select a Stock Code" and graphs
with col2:
    # Input: Stock code from the top 10 products
    selected_stock_code = st.selectbox("Select a Stock Code:", top_10_products)

    # Filter data for the selected stock code
    selected_data = weekly_top_10_with_stockcode[weekly_top_10_with_stockcode['StockCode'] == selected_stock_code]

    # ARIMA model for forecasting
    st.subheader("Historical and Forecast Plot")
    try:
        arima_model = ARIMA(selected_data['Quantity'], order=(5, 1, 0))
        arima_model_fit = arima_model.fit()
        forecast = arima_model_fit.forecast(steps=15)
        forecast = np.maximum(forecast, 0)  # Ensure forecast is non-negative

        # Historical and Forecast Plot
        fig, ax = plt.subplots()
        ax.plot(selected_data.index, selected_data['Quantity'], label='Historical')
        ax.plot(pd.date_range(start=selected_data.index[-1], periods=15, freq='W'), forecast, label='Forecast')
        ax.legend()
        st.pyplot(fig)
    except np.linalg.LinAlgError:
        st.error("The data is not suitable for the ARIMA model (e.g., insufficient data points, non-stationary data, etc.)")
    except Exception as e:
        st.error(f"Error in ARIMA model fitting: {e}") #the data is not suitable for the ARIMA model (e.g., insufficient data points, non-stationary data, etc.).

    # Actual vs Predicted Plot
    st.subheader("Actual vs Predicted Demand")
    fig, ax = plt.subplots()
    ax.plot(weekly_top_10_with_stockcode.index, weekly_top_10_with_stockcode['Revenue'], label='Actual')
    ax.plot(weekly_top_10_with_stockcode.index[-len(dt_predictions_all):], dt_predictions_all, label='DecisionTree Predicted')
    ax.plot(weekly_top_10_with_stockcode.index[-len(xgb_predictions_all):], xgb_predictions_all, label='XGBoost Predicted')
    ax.legend()
    st.pyplot(fig)

    # Error Histogram
    st.subheader("Error Histogram")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].hist(np.array(actual_all) - np.array(dt_predictions_all), bins=20, alpha=0.7, label='DecisionTree')
    ax[0].set_title('DecisionTree Errors')
    ax[1].hist(np.array(actual_all) - np.array(xgb_predictions_all), bins=20, alpha=0.7, label='XGBoost')
    ax[1].set_title('XGBoost Errors')
    st.pyplot(fig)

        # Calculate the maximum allowable number of lags
    max_lags = len(weekly_top_10) // 2 - 1

    # Plot ACF and PACF
    st.subheader("ACF and PACF Plots")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    plot_acf(weekly_top_10_with_stockcode['Quantity'], lags=max_lags, ax=ax[0])
    plot_pacf(weekly_top_10_with_stockcode['Quantity'], lags=max_lags, ax=ax[1])
    st.pyplot(fig)