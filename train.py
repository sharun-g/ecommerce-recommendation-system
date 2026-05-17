# Import required modules and functions
from src.data_loader import load_data
from src.preprocess import preprocess
from src.fp_growth_model import run_fp_growth
from src.cf_model import build_cf

import pickle

print("Loading data...")

# Load datasets
orders, order_products, products = load_data()

print("Preprocessing...")

# Clean and prepare transaction data
data = preprocess(orders, order_products)

# Save processed dataset
data.to_csv("data/processed_data.csv", index=False)

print("Running FP-Growth...")

# Generate association rules using FP-Growth
rules = run_fp_growth(data)

# Save FP-Growth rules
rules.to_pickle("data/fp_rules.pkl")

print("Building CF model...")

# Build Collaborative Filtering similarity matrix
similarity, product_ids = build_cf(data)

# Save similarity matrix
with open("data/similarity.pkl", "wb") as f:
    pickle.dump(similarity, f)

# Save product ID mapping
with open("data/product_ids.pkl", "wb") as f:
    pickle.dump(product_ids, f)

print("Training complete & saved!")
