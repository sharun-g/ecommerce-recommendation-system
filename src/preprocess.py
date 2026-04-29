import pandas as pd

def preprocess(orders, order_products, user_limit=20000, product_limit=1000):

    top_users = orders['user_id'].value_counts().head(user_limit).index
    orders = orders[orders['user_id'].isin(top_users)]

    data = pd.merge(order_products, orders, on="order_id")

    top_products = data['product_id'].value_counts().head(product_limit).index
    data = data[data['product_id'].isin(top_products)]

    return data