from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity

def build_cf(data):

    user_ids = data['user_id'].astype("category").cat.codes
    product_ids = data['product_id'].astype("category").cat.codes

    matrix = coo_matrix(
        ([1]*len(data), (user_ids, product_ids))
    )

    similarity = cosine_similarity(matrix.T, dense_output=False)

    return similarity, product_ids