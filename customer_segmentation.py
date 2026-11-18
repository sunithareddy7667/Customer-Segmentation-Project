# CUSTOMER SEGMENTATION PROJECT

# Step 1: Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

print("Libraries imported successfully!")


# Step 2: Load Dataset

df = pd.read_excel("dataset/Online Retail.xlsx",engine="openpyxl")

print("\nDataset Loaded Successfully!")

# Display first 5 records
print("\nFirst Five Records:")
print(df.head())


# Step 3: Dataset Information

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Columns:")
print(df.columns)

print("\nDataset Information:")
print(df.info())
# STEP 4: DATA CLEANING

print("\nStarting Data Cleaning...")

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Remove duplicate records
df = df.drop_duplicates()

# Remove missing CustomerID
df = df.dropna(subset=['CustomerID'])

# Remove cancelled transactions
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

# Remove negative and zero quantities
df = df[df['Quantity'] > 0]

# Remove zero and negative prices
df = df[df['UnitPrice'] > 0]

# Calculate total purchase amount
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

print("\nData Cleaning Completed!")

print("\nCleaned Dataset Shape:")
print(df.shape)

print("\nCleaned Dataset:")
print(df.head())

# Save cleaned dataset
df.to_csv("outputs/cleaned_retail.csv", index=False)

print("\nCleaned dataset saved successfully!")
# STEP 5: EXPLORATORY DATA ANALYSIS

print("\n========== EXPLORATORY DATA ANALYSIS ==========")

# Total Revenue
total_revenue = df['TotalPrice'].sum()

print("\nTotal Revenue:")
print(round(total_revenue, 2))

# Total Customers
total_customers = df['CustomerID'].nunique()

print("\nTotal Customers:")
print(total_customers)

# Total Transactions
total_transactions = df['InvoiceNo'].nunique()

print("\nTotal Transactions:")
print(total_transactions)

# Top 10 Products
top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)

print("\nTop 10 Products:")
print(top_products)

# Sales by Country
country_sales = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)

print("\nTop 10 Countries by Sales:")
print(country_sales)


# VISUALIZATION 1: Top 10 Products

plt.figure(figsize=(12, 6))

top_products.plot(kind='bar')

plt.title("Top 10 Most Purchased Products")
plt.xlabel("Products")
plt.ylabel("Quantity Sold")
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig("outputs/top_products.png")
plt.show()


# VISUALIZATION 2: Sales by Country

plt.figure(figsize=(12, 6))

country_sales.plot(kind='bar')

plt.title("Top 10 Countries by Revenue")
plt.xlabel("Country")
plt.ylabel("Revenue")

plt.tight_layout()
plt.savefig("outputs/country_sales.png")
plt.show()


print("\nEDA Completed Successfully!")

# STEP 6: RFM ANALYSIS

print("\nStarting RFM Analysis...")

# Read cleaned dataset
df = pd.read_csv("outputs/cleaned_retail.csv")

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Calculate reference date
reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# Create RFM table
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
})

# Rename columns
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Display RFM results
print("\nRFM Analysis Results:")
print(rfm.head(10))

# Save RFM dataset
rfm.to_csv("outputs/rfm_analysis.csv")

print("\nRFM Analysis Completed Successfully!")
print("RFM dataset saved successfully!")
# Step 5: Customer Segmentation using K-Means

print("\nStarting Customer Segmentation...")

# Select RFM features
X = rfm[['Recency', 'Frequency', 'Monetary']]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)

rfm['Cluster'] = kmeans.fit_predict(X_scaled)

# Display customer segments
print("\nCustomer Segmentation Results:")
print(rfm.head(10))

# Count customers in each cluster
print("\nCustomers in Each Segment:")
print(rfm['Cluster'].value_counts())

# Calculate cluster characteristics
cluster_summary = rfm.groupby('Cluster')[
    ['Recency', 'Frequency', 'Monetary']
].mean()

print("\nCluster Summary:")
print(cluster_summary)

# Save segmentation results
rfm.to_csv("outputs/customer_segments.csv")

print("\nCustomer Segmentation Completed Successfully!")
print("Results saved successfully!")
# Step 6: Customer Segmentation Visualization

import matplotlib.pyplot as plt
import seaborn as sns

print("\nCreating Customer Segmentation Graphs...")

# Graph 1: Number of customers in each cluster

plt.figure(figsize=(8, 5))

sns.countplot(data=rfm, x='Cluster')

plt.title("Customer Distribution by Segment")
plt.xlabel("Customer Cluster")
plt.ylabel("Number of Customers")

plt.savefig("outputs/customer_distribution.png")
plt.show()


# Graph 2: Monetary Value by Cluster

plt.figure(figsize=(8, 5))

sns.barplot(
    data=rfm,
    x='Cluster',
    y='Monetary',
    errorbar=None
)

plt.title("Average Spending by Customer Segment")
plt.xlabel("Customer Cluster")
plt.ylabel("Average Monetary Value")

plt.savefig("outputs/customer_spending.png")
plt.show()


# Graph 3: Frequency vs Monetary

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=rfm,
    x='Frequency',
    y='Monetary',
    hue='Cluster',
    palette='viridis'
)

plt.title("Customer Segmentation Analysis")
plt.xlabel("Purchase Frequency")
plt.ylabel("Monetary Value")

plt.savefig("outputs/customer_segments_graph.png")
plt.show()

print("\nAll Customer Segmentation Graphs Created Successfully!")