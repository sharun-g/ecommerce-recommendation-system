from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import pandas as pd

def run_fp_growth(data):

    transactions = data.groupby("order_id")["product_id"].apply(list)

    te = TransactionEncoder()
    te_data = te.fit(transactions).transform(transactions)

    df = pd.DataFrame(te_data, columns=te.columns_)

    frequent_itemsets = fpgrowth(df, min_support=0.02, use_colnames=True)

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

    rules = rules[
        (rules['antecedents'].apply(len) == 1) &
        (rules['consequents'].apply(len) == 1)
    ]

    return rules