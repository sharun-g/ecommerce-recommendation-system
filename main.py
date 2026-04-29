import pandas as pd
import pickle
from src.preprocess import preprocess
from src.data_loader import load_data
from src.hybrid_model import build_fp_dict, hybrid_recommend

from evaluation import fast_precision_at_k

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

########## --this will take time to calculate the score of the models which represents avrage recommendation strength of models
data = preprocess(orders, order_products)
data_sample = data.sample(30000)

def cf_wrapper(item, top_n=5):
    # use your CF-only recommendation logic
    return hybrid_recommend(item, data, similarity, product_ids, {}, alpha=1.0, top_n=top_n)

cf_score = fast_precision_at_k(data_sample, cf_wrapper)
print("CF Score:", cf_score)

def fp_wrapper(item, top_n=5):
    return hybrid_recommend(item, data, similarity, product_ids, fp_dict, alpha=0.0, top_n=top_n)

fp_score = fast_precision_at_k(data_sample, fp_wrapper)
print("FP Score:", fp_score)

def hybrid_wrapper(item, top_n=5):
    return hybrid_recommend(item, data, similarity, product_ids, fp_dict, alpha=0.7, top_n=top_n)

hybrid_score = fast_precision_at_k(data_sample, hybrid_wrapper)
print("Hybrid Score:", hybrid_score)
######
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