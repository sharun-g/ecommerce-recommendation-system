import numpy as np

def build_fp_dict(rules):
    fp_dict = {}

    for _, row in rules.iterrows():
        a = list(row['antecedents'])[0]
        b = list(row['consequents'])[0]

        if a not in fp_dict:
            fp_dict[a] = []

        fp_dict[a].append((b, row['confidence']))
        
    return fp_dict


def hybrid_recommend(product_id, data, similarity, product_ids, fp_dict, alpha=0.7, top_n=5):

    # Map product_id → internal index
    prod_map = dict(zip(data['product_id'], product_ids))
    reverse_map = dict(enumerate(data['product_id'].astype("category").cat.categories))

    pid = prod_map[product_id]

    # CF scores
    cf_scores = similarity[pid].toarray().flatten()
    cf_scores = cf_scores / (cf_scores.max() + 1e-9)

    final_scores = {}

    # CF contribution
    for idx, score in enumerate(cf_scores):
        final_scores[idx] = alpha * score

    # FP contribution
    if product_id in fp_dict:
        for item, conf in fp_dict[product_id]:
            if item in prod_map:
                idx = prod_map[item]
                final_scores[idx] += (1 - alpha) * conf

    # Rank
    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    ranked = [r for r in ranked if reverse_map[r[0]] != product_id]

    results = [reverse_map[r[0]] for r in ranked[:top_n]]

    return results