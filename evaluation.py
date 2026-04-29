import random

def fast_precision_at_k(data, model_func, k=5, user_sample=200):

    hits = 0
    total = 0

    # Group user purchases
    user_groups = data.groupby("user_id")["product_id"].apply(list)

    # Sample users
    sampled_users = random.sample(list(user_groups.index), min(user_sample, len(user_groups)))

    for user in sampled_users:
        items = user_groups[user]

        # Take only 1 item per user (VERY IMPORTANT)
        item = random.choice(items)

        actual = set(items)

        try:
            recs = model_func(item, top_n=k)

            hit = len(set(recs) & actual)

            hits += hit
            total += k

        except:
            continue

    return hits / total if total > 0 else 0