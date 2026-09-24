import math


def evaluate_rankings(rankings, relevant_docs):
    """
    rankings:
        {
            "q1": ["d10", "d1", "d7", ...],
            ...
        }

    relevant_docs:
        {
            "q1": "d1",
            ...
        }
    """

    ndcg10_scores = []
    reciprocal_ranks = []

    recall_10 = []
    recall_20 = []
    recall_100 = []

    for query_id, correct_doc in relevant_docs.items():
        ranked_docs = rankings[query_id]

        try:
            rank = ranked_docs.index(correct_doc) + 1
        except ValueError:
            rank = None

        # NDCG@10
        if rank is not None and rank <= 10:
            ndcg10 = 1.0 / math.log2(rank + 1)
        else:
            ndcg10 = 0.0

        ndcg10_scores.append(ndcg10)

        # MRR
        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        recall_10.append(
            1.0 if correct_doc in ranked_docs[:10] else 0.0
        )

        recall_20.append(
            1.0 if correct_doc in ranked_docs[:20] else 0.0
        )

        recall_100.append(
            1.0 if correct_doc in ranked_docs[:100] else 0.0
        )

    n = len(relevant_docs)

    return {
        "ndcg@10": sum(ndcg10_scores) / n,
        "mrr": sum(reciprocal_ranks) / n,
        "recall@10": sum(recall_10) / n,
        "recall@20": sum(recall_20) / n,
        "recall@100": sum(recall_100) / n,
    }
