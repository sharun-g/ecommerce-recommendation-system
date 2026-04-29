import pandas as pd

def load_data():
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

    return orders, order_products, products