import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

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

# sort the transactional data by 'Revenue' in ascending order
transactional_data.sort_values(by='Revenue', ascending=False, inplace=True)

# Prepare the data: Combine customer demographic data with transactional data
# Merge transactional data with customer demographic info
customer_demographic_productinfo = pd.merge(customer_demographic, transactional_data, on='Customer ID', how='inner')
# in the description column, there are some nan values, so we will fill them with 'unknown'.
customer_demographic_productinfo['Description'] = customer_demographic_productinfo['Description'].fillna('unknown')
# Exploratory Data Analysis (EDA):
# 1. Perform customer, item, and transaction-level summary statistics.
# 2. Utilize SQL join queries to retrieve necessary data (e.g., customer and product
# information) without explicit data merging.
# 3. Consolidate transactional data where necessary to ensure accurate summary metrics for
# each level (customer, item, transaction).
# 4. Design and develop visualizations which should help to explain the data and summary
# statistics.
# 1. Perform customer, item, and transaction-level summary statistics.
# Customer-level summary statistics
customer_summary = customer_demographic_productinfo.groupby('Customer ID').agg({
    'Quantity': 'sum',
    'Price': 'sum',
    'Invoice': 'nunique'  # Number of unique transactions per customer
}).rename(columns={'Quantity': 'Total_Quantity', 'Price': 'Total_Price', 'Invoice': 'Num_Transactions'})

customer_summary_index = customer_summary.reset_index()

# 2. Item (Product) level summary statistics
item_summary = customer_demographic_productinfo.groupby('StockCode').agg({
    'Quantity': 'sum',
    'Price': 'mean'  # Average price for each item
}).rename(columns={'Quantity': 'Total_Quantity', 'Price': 'Avg_Price'})
item_summary_index = item_summary.reset_index()

# 3. Transaction-level summary statistics
transaction_summary = customer_demographic_productinfo.groupby('Invoice').agg({
    'Quantity': 'sum',
    'Price': 'sum'
}).rename(columns={'Quantity': 'Total_Quantity', 'Price': 'Total_Price'})
transaction_summary_index = transaction_summary.reset_index()

# customer_summary, item_summary, and transaction_summary

# Bar plot for Customer Summary
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
sns.barplot(x=customer_summary_index.index[:10], y=customer_summary_index['Total_Quantity'][:10])
plt.title('Total Quantity per Customer')
plt.xlabel('Customer ID')
plt.ylabel('Total Quantity')
plt.xticks(rotation=45)

plt.subplot(1, 3, 2)
sns.barplot(x=customer_summary_index.index[:10], y=customer_summary_index['Total_Price'][:10])
plt.title('Total Price per Customer')
plt.xlabel('Customer ID')
plt.ylabel('Total Price')
plt.xticks(rotation=45)

plt.subplot(1, 3, 3)
sns.barplot(x=customer_summary_index.index[:10], y=customer_summary_index['Num_Transactions'][:10])
plt.title('Number of Transactions per Customer')
plt.xlabel('Customer ID')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# Bar plot for Item Summary
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
sns.barplot(x=item_summary_index.index[:10], y=item_summary_index['Total_Quantity'][:10])
plt.title('Total Quantity per Item')
plt.xlabel('StockCode')
plt.ylabel('Total Quantity')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.barplot(x=item_summary_index.index[:10], y=item_summary_index['Avg_Price'][:10])
plt.title('Average Price per Item')
plt.xlabel('StockCode')
plt.ylabel('Average Price')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Bar plot for Transaction Summary
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.barplot(x=transaction_summary_index.index[:10], y=transaction_summary_index['Total_Quantity'][:10])
plt.title('Total Quantity per Transaction')
plt.xlabel('Invoice')
plt.ylabel('Total Quantity')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.barplot(x=transaction_summary_index.index[:10], y=transaction_summary_index['Total_Price'][:10])
plt.title('Total Price per Transaction')
plt.xlabel('Invoice')
plt.ylabel('Total Price')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# 2. Utilize SQL join queries to retrieve necessary data 
# (e.g., customer and product information) without explicit data merging
# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('data/retail_data.db')

# Create a cursor object
cursor = conn.cursor()

# Create customer_demographics table
customer_demographic.to_sql('customer_demographics', conn, index=False, if_exists='replace')

# Create transactional_data_retail_01 table
transactional_data_01.to_sql('transactional_data_retail_01', conn, index=False, if_exists='replace')

# Create transactional_data_retail_02 table
transactional_data_02.to_sql('transactional_data_retail_02', conn, index=False, if_exists='replace')

# Create productsinfo table
productsinfo.to_sql('productinfo', conn, index=False, if_exists='replace')

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())

cursor.execute("PRAGMA table_info(customer_demographics);")
column_names = cursor.fetchall()
for column in column_names:
  print(column[1])

# PRAGMA is a command used in SQLite to modify the operation of the SQLite database engine.
# It's a way to interact with the database itself, rather than with the data within the tables.

# 1. Joining Transactional Data Retail 01 with Customer Demographic.

query = '''
CREATE TABLE transaction_customer_data_customer_demographics_product AS
SELECT
transcation_customer_data_customer_demographics."Customer ID" AS "Customer_ID",
transcation_customer_data_customer_demographics."Country",
transcation_customer_data_customer_demographics."Invoice",
transcation_customer_data_customer_demographics."InvoiceDate",
transcation_customer_data_customer_demographics."Quantity",
transcation_customer_data_customer_demographics."Price",
transcation_customer_data_customer_demographics."StockCode" AS "Stock_Code",
productinfo."Description"
FROM 
(SELECT * FROM 
(SELECT * FROM transactional_data_retail_01
UNION
SELECT * FROM transactional_data_retail_02) AS transaction_customer_data
INNER JOIN customer_demographics
ON transaction_customer_data."Customer ID" = customer_demographics."Customer ID") AS transcation_customer_data_customer_demographics
INNER JOIN productinfo
ON transcation_customer_data_customer_demographics."StockCode" = productinfo."StockCode";
'''

cursor.execute(query)

# Execute the PRAGMA command to get column info
cursor.execute("PRAGMA table_info(transaction_customer_data_customer_demographics_product);")

# Fetch all results
columns_info = cursor.fetchall()
# Extract and print column names
column_names = [column[1] for column in columns_info]  # The second item in each tuple is the column name
print(column_names)


# 3. Consolidate transactional data where necessary to ensure accurate summary metrics for each level (customer, item, transaction).
# Consolidate data at customer level
query_1 = '''
CREATE TABLE customer_summary AS
SELECT
    "Customer_ID",
    COUNT(DISTINCT "Invoice") AS total_transactions,
    SUM("Quantity") AS total_quantity,
    SUM("Price" * "Quantity") AS total_sales
FROM
    transaction_customer_data_customer_demographics_product
GROUP BY
    "Customer_ID";

'''

# Consolidate data by product level

query_2 = ''' CREATE TABLE item_summary AS
SELECT
    "Stock_Code",
    COUNT(DISTINCT "Invoice") AS total_transactions,
    SUM("Quantity") AS total_quantity,
    SUM("Price" * "Quantity") AS total_sales
FROM
    transaction_customer_data_customer_demographics_product
GROUP BY
    "Stock_Code";

'''

# Consolidate data by Transaction level

query_3 = '''CREATE TABLE transaction_summary AS
SELECT
    "Invoice", "InvoiceDate",
    COUNT("Stock_Code") AS item_count,
    SUM("Quantity") AS total_quantity,
    SUM("Price" * "Quantity") AS total_sales
FROM
    transaction_customer_data_customer_demographics_product
GROUP BY
    "Invoice";

'''
cursor.execute(query_1)
cursor.execute(query_2)
cursor.execute(query_3)

cursor.execute("PRAGMA table_info(customer_summary);")
column_names = cursor.fetchall()
for column in column_names:
  print(column[1])


# 4. Design and develop visualizations which should help to explain the data and summary statistics.
df_customer_summary = pd.read_sql_query("SELECT * FROM customer_summary;", conn)
df_item_summary = pd.read_sql_query("SELECT * FROM item_summary;", conn)
df_transaction_summary = pd.read_sql_query("SELECT * FROM transaction_summary;", conn)

# 1. Customer Level Visulization.
# Top 10 Cutomers by Total Sales.
plt.figure(figsize=(10,6))
top_customers = df_customer_summary.sort_values('total_sales', ascending=False).head(10)
sns.barplot(x="Customer_ID", y="total_sales", data=top_customers)
plt.title('Top 10 Customers by Total Sales')
plt.xticks(rotation=45)
plt.ylabel('Total Sales')
plt.show()

# 2. Item-Level Visualizations
# Top 10 Selling Product by Quantity
plt.figure(figsize=(10,6))
top_items = df_item_summary.sort_values('total_quantity', ascending=False).head(10)
sns.barplot(x="Stock_Code", y="total_quantity", data=top_items)
plt.title('Top 10 Stock Codes Based on Total Quantity Sold')
plt.xticks(rotation=45)
plt.ylabel('Total Quantity Sold')
plt.show()


# Top 10 Products by Sales Share
plt.figure(figsize=(8,8))
top_items_sales = df_item_summary.sort_values('total_sales', ascending=False).head(10)
plt.pie(top_items_sales['total_sales'], labels=top_items_sales['Stock_Code'], autopct='%1.1f%%')
plt.title('Top 10 High Revenue Products')
plt.show()

# 3. Transaction-Level Visualizations
df_transaction_summary['InvoiceDate'] = pd.to_datetime(df_transaction_summary['InvoiceDate'], format='%d-%m-%Y', errors='coerce')
df_transaction_summary.set_index('InvoiceDate', inplace=True)

# Resampling by week or month for smooth trend
weekly_sales = df_transaction_summary.resample('W').sum()

plt.figure(figsize=(10,6))
plt.plot(weekly_sales.index, weekly_sales['total_sales'], marker='o')
plt.title('Total Sales Over Time (Weekly)')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(x=df_transaction_summary['total_sales'])
plt.title('Distribution of Sales per Transaction')
plt.xlabel('Total Sales')
plt.show()
# Select unique products
# top_selling_products = transactional_data.drop_duplicates(subset=['StockCode']).reset_index(drop=True)

# # Top 10 best selling products
# top_10_selling_products = top_selling_products.head(10)

# visualize the top 10 selling products


# plt.figure(figsize=(10, 6))
# sns.barplot(x='Revenue', y='StockCode', data=top_10_selling_products, palette='viridis')
# plt.xlabel('Revenue')
# plt.ylabel('StockCode')
# plt.title('Top 10 Selling Products')

# # Add descriptions as text labels on the bars
# for index, value in enumerate(top_10_selling_products['Revenue']):
#     plt.text(value, index, str(top_10_selling_products['Description'][index]), color='black', ha="left")

# plt.show()

# Non-Time Series Techniques: Apply machine learning models (e.g., DecisionTree, XGBoost) that 
# leverage non-time series features, such as customer demographics and product features, to 
# predict demand. 

# Prepare the data: Combine customer demographic data with transactional data
# Merge transactional data with customer demographic info
# customer_demographic_productinfo = pd.merge(customer_demographic, transactional_data, on='Customer ID', how='inner')


# # separte the target variable from the features
# X = customer_demographic_productinfo.drop(['Revenue', 'InvoiceDate', 'Description', 'Price', 'Quantity','Customer ID'], axis=1)
# y = customer_demographic_productinfo['Revenue']

# # Convert categorical variables to numerical
# encoder = OneHotEncoder(sparse_output=False, drop='first')
# categorical_columns = X.select_dtypes(include=['object']).columns

# ct = ColumnTransformer(transformers=[('encoder', encoder, categorical_columns)], remainder='passthrough', verbose_feature_names_out=False).set_output(transform='pandas')
# X_encoded = ct.fit_transform(X)

# # Define the time-based cross-validation strategy
# tscv = TimeSeriesSplit(n_splits=5)

# # Train and evaluate the models using time-based cross-validation
# dt_rmse_scores = []
# dt_mae_scores = []
# xgb_rmse_scores = []
# xgb_mae_scores = []

# for train_index, test_index in tscv.split(X):
#     X_train, X_test = X_encoded.iloc[train_index], X_encoded.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
#     # Train a DecisionTree model
#     dt_model = DecisionTreeRegressor()
#     dt_model.fit(X_train, y_train)
#     dt_predictions = dt_model.predict(X_test)
#     dt_rmse_scores.append(root_mean_squared_error(y_test, dt_predictions))
#     dt_mae_scores.append(mean_absolute_error(y_test, dt_predictions))
    
#     # Train an XGBoost model
#     xgb_model = XGBRegressor()
#     xgb_model.fit(X_train, y_train)
#     xgb_predictions = xgb_model.predict(X_test)
#     xgb_rmse_scores.append(root_mean_squared_error(y_test, xgb_predictions))
#     xgb_mae_scores.append(mean_absolute_error(y_test, xgb_predictions))

# print(f"DecisionTree - RMSE: {dt_rmse_scores}, MAE: {dt_mae_scores}")
# print(f"XGBoost - RMSE: {xgb_rmse_scores}, MAE: {xgb_mae_scores}")

# # Feature importance for DecisionTree
# feature_importances = pd.Series(dt_model.feature_importances_, index=X_encoded.columns)
# feature_importances.nlargest(10).plot(kind='barh')
# plt.show()

# # Feature importance for XGBoost
# feature_importances = pd.Series(xgb_model.feature_importances_, index=X_encoded.columns)
# feature_importances.nlargest(10).plot(kind='barh')
# plt.show()
