import pandas as pd
import pickle
from src.data_loader import load_data
from src.hybrid_model import build_fp_dict, hybrid_recommend

print("Loading saved data...")

data = pd.read_csv("data/processed_data.csv")

rules = pd.read_pickle("data/fp_rules.pkl")


with open("data/similarity.pkl", "rb") as f:
    similarity = pickle.load(f)

with open("data/product_ids.pkl", "rb") as f:
    product_ids = pickle.load(f)


orders, order_products, products = load_data()

valid_products = set(data['product_id'].unique())

name_to_id = dict(zip(products['product_name'].str.lower(), products['product_id']))
id_to_name = dict(zip(products['product_id'], products['product_name']))

fp_dict = build_fp_dict(rules)

print("\nSystem Ready")

while True:
    user_input = input("\nEnter product name (or 'exit'): ").lower().strip()
    
    if user_input == "exit":
        break

    if user_input not in name_to_id:
        print("Product not found")
        continue

    product_id = name_to_id[user_input]

    if product_id not in valid_products:
        print("This product is not in the trained model.")
        continue
    

    recs = hybrid_recommend(
        product_id,
        data,
        similarity,
        product_ids,
        fp_dict
    )

    rec_names = [id_to_name.get(i, "Unknown") for i in recs]

    print("\nRecommended Products:\n")
    for r in rec_names:
        print(r)