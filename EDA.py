import pandas as pd
import matplotlib.pyplot as plt

orders = pd.read_csv(
    "data/orders.csv",
    usecols=["order_id", "user_id"],
    dtype={"order_id": "int32", "user_id": "int32"}
)

order_products = pd.read_csv(
    "data/order_products__prior.csv",
    usecols=["order_id", "product_id"],
    dtype={"order_id": "int32", "product_id": "int32"}
)

products = pd.read_csv(
    "data/products.csv",
    usecols=["product_id", "product_name"]
)


print("---Basic Info:---")
print("Total Orders:", orders.shape[0])
print("Total Users:", orders['user_id'].nunique())
print("Total Products:", products.shape[0])
print()

print("--NULL VALUE CHECK:--\n")

print("Orders:")
print(orders.isnull().sum(), "\n")

print("Order Products:")
print(order_products.isnull().sum(), "\n")

print("Products:")
print(products.isnull().sum(), "\n")

sample_orders = orders.sample(50000, random_state=42)
data = pd.merge(order_products, sample_orders, on="order_id")

#Top Products
top_products = data['product_id'].value_counts().head(10)

top_products = top_products.reset_index()
top_products.columns = ['product_id', 'count']

top_products = top_products.merge(products, on="product_id")

print("---Top 10 Products:---")
print(top_products[['product_name', 'count']])
print()

