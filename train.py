from src.data_loader import load_data
from src.preprocess import preprocess
from src.fp_growth_model import run_fp_growth
from src.cf_model import build_cf

import pickle

print("Loading data...")
orders, order_products, products = load_data()

print("Preprocessing...")
data = preprocess(orders, order_products)

data.to_csv("data/processed_data.csv", index=False)

print("Running FP-Growth...")
rules = run_fp_growth(data)
rules.to_pickle("data/fp_rules.pkl")

print("Building CF model...")
similarity, product_ids = build_cf(data)

with open("data/similarity.pkl", "wb") as f:
    pickle.dump(similarity, f)

# Save product_ids mapping too
with open("data/product_ids.pkl", "wb") as f:
    pickle.dump(product_ids, f)

print("Training complete & saved!")