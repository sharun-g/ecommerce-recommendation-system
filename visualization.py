import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import load_data

orders, order_products, products = load_data()

top_products = order_products['product_id'].value_counts().head(10).reset_index()
top_products.columns = ['product_id', 'count']

top_products = top_products.merge(products, on="product_id")

plt.figure()
plt.barh(top_products['product_name'], top_products['count'])
plt.title("Top 10 Most Purchased Products")
plt.xlabel("Count")
plt.ylabel("Product")
plt.gca().invert_yaxis()
plt.show()

top_users = orders['user_id'].value_counts().head(10)

plt.figure()
plt.bar(range(len(top_users)), top_users.values)
plt.title("Top 10 Active Users")
plt.xlabel("User Index")
plt.ylabel("Orders")
plt.show()

orders_per_user = orders['user_id'].value_counts()

plt.figure()
plt.hist(orders_per_user, bins=50)
plt.title("Orders per User Distribution")
plt.xlabel("Orders")
plt.ylabel("Frequency")
plt.show()

product_counts = order_products['product_id'].value_counts()

plt.figure()
plt.hist(product_counts, bins=50)
plt.title("Product Frequency Distribution")
plt.xlabel("Purchases")
plt.ylabel("Products")
plt.show()

models = ['CF', 'FP-Growth', 'Hybrid']
scores = [0.009, 0.003, 0.014]

plt.figure()
plt.bar(models, scores)
plt.title("Model Comparison (Precision@5)")
plt.xlabel("Model")
plt.ylabel("Score")
plt.show()