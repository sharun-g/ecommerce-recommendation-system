import pandas as pd
import pickle

# Import required functions
from src.data_loader import load_data
from src.hybrid_model import build_fp_dict, hybrid_recommend

print("Loading saved data...")

# Load processed transaction data
data = pd.read_csv("data/processed_data.csv")

# Load FP-Growth association rules
rules = pd.read_pickle("data/fp_rules.pkl")

# Load Collaborative Filtering similarity matrix
with open("data/similarity.pkl", "rb") as f:
    similarity = pickle.load(f)

# Load trained product IDs
with open("data/product_ids.pkl", "rb") as f:
    product_ids = pickle.load(f)

# Load original datasets
orders, order_products, products = load_data()

# Store products available in trained model
valid_products = set(data['product_id'].unique())

# Mapping: product name -> product ID
name_to_id = dict(
    zip(products['product_name'].str.lower(), products['product_id'])
)

# Mapping: product ID -> product name
id_to_name = dict(
    zip(products['product_id'], products['product_name'])
)

# Build FP-Growth recommendation dictionary
fp_dict = build_fp_dict(rules)

print("\nSystem Ready")

# Main recommendation loop
while True:

    # Take product name input from user
    user_input = input(
        "\nEnter product name (or 'exit'): "
    ).lower().strip()

    # Exit condition
    if user_input == "exit":
        break

    # Check if product exists
    if user_input not in name_to_id:
        print("Product not found")
        continue

    # Get product ID
    product_id = name_to_id[user_input]

    # Check if product exists in trained dataset
    if product_id not in valid_products:
        print("This product is not in the trained model.")
        continue

    # Generate hybrid recommendations
    recs = hybrid_recommend(
        product_id,
        data,
        similarity,
        product_ids,
        fp_dict
    )

    # Convert recommended product IDs to product names
    rec_names = [
        id_to_name.get(i, "Unknown")
        for i in recs
    ]

    # Display recommendations
    print("\nRecommended Products:\n")

    for r in rec_names:
        print(r)
