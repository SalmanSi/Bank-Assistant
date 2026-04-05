"""
Retrieval Evaluation Script for Bank Assistant RAG Pipeline.

Evaluates the retrieval step using standard IR metrics:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG@K)
- Hit Rate@K

Usage:
    PYTHONPATH=. uv run python eval/evaluate_retrieval.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.build_vectordb import (
    get_embedding_model,
    load_vectorstore,
    query_vectorstore,
)
from eval.eval_ground_truth import GROUND_TRUTH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [EVAL] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "data" / "eval_results.json"


def dcg_at_k(relevances: list[bool], k: int) -> float:
    """Compute DCG@k."""
    relevances = relevances[:k]
    return sum((1 / np.log2(i + 2)) if rel else 0 for i, rel in enumerate(relevances))


def ndcg_at_k(relevances: list[bool], k: int) -> float:
    """Compute NDCG@k."""
    dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute Precision@K."""
    retrieved_k = retrieved_ids[:k]
    if not retrieved_k:
        return 0.0
    return sum(1 for rid in retrieved_k if rid in relevant_ids) / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute Recall@K."""
    retrieved_k = retrieved_ids[:k]
    if not relevant_ids:
        return 0.0
    return sum(1 for rid in retrieved_k if rid in relevant_ids) / len(relevant_ids)


def hit_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> bool:
    """Check if there's at least one hit in top-k."""
    return any(rid in relevant_ids for rid in retrieved_ids[:k])


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Compute reciprocal rank (1/rank of first relevant doc)."""
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


def average_precision(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Compute Average Precision."""
    if not relevant_ids:
        return 0.0

    score = 0.0
    num_hits = 0
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_ids:
            num_hits += 1
            score += num_hits / i
    return score / len(relevant_ids)


def get_retrieved_parent_ids(results: list[dict[str, Any]]) -> list[str]:
    """Extract parent_ids from retrieval results."""
    return [r.get("parent_id", "") for r in results if r.get("parent_id")]


def evaluate_retrieval(
    collection: Any,
    model: Any,
    ground_truth: list[dict],
    top_k: int = 10,
) -> dict[str, Any]:
    """Run full evaluation over ground truth queries."""
    k_values = [1, 3, 5, 10]

    metrics = {f"precision@{k}": [] for k in k_values}
    metrics.update({f"recall@{k}": [] for k in k_values})
    metrics.update({f"ndcg@{k}": [] for k in k_values})
    metrics.update({f"hit@{k}": [] for k in k_values})
    metrics["mrr"] = []
    metrics["map"] = []

    detailed_results = []

    for item in ground_truth:
        query = item["query"]
        relevant_ids = item["relevant_ids"]

        try:
            results = query_vectorstore(
                query,
                collection=collection,
                model=model,
                top_k=top_k,
            )
        except Exception as e:
            logger.error("Query failed: %s - %s", query, e)
            continue

        retrieved_ids = get_retrieved_parent_ids(results)
        relevances = [rid in relevant_ids for rid in retrieved_ids[:top_k]]

        query_metrics = {"query": query, "relevant_count": len(relevant_ids)}

        for k in k_values:
            query_metrics[f"precision@{k}"] = precision_at_k(
                retrieved_ids, relevant_ids, k
            )
            query_metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
            query_metrics[f"ndcg@{k}"] = ndcg_at_k(relevances, k)
            query_metrics[f"hit@{k}"] = (
                1.0 if hit_at_k(retrieved_ids, relevant_ids, k) else 0.0
            )

        query_metrics["mrr"] = reciprocal_rank(retrieved_ids, relevant_ids)
        query_metrics["map"] = average_precision(retrieved_ids, relevant_ids)

        for k in k_values:
            metrics[f"precision@{k}"].append(query_metrics[f"precision@{k}"])
            metrics[f"recall@{k}"].append(query_metrics[f"recall@{k}"])
            metrics[f"ndcg@{k}"].append(query_metrics[f"ndcg@{k}"])
            metrics[f"hit@{k}"].append(query_metrics[f"hit@{k}"])

        metrics["mrr"].append(query_metrics["mrr"])
        metrics["map"].append(query_metrics["map"])

        query_metrics["retrieved_ids"] = retrieved_ids[:5]
        detailed_results.append(query_metrics)

    summary = {}
    for metric_name, values in metrics.items():
        if values:
            summary[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    return {
        "summary": summary,
        "detailed_results": detailed_results,
        "num_queries": len(detailed_results),
        "top_k": top_k,
    }


def analyze_by_category(
    collection: Any,
    model: Any,
    ground_truth: list[dict],
    top_k: int = 10,
) -> dict[str, Any]:
    """Analyze performance grouped by query categories."""
    category_results: dict[str, list[dict]] = {}

    for item in ground_truth:
        results = query_vectorstore(
            item["query"],
            collection=collection,
            model=model,
            top_k=top_k,
        )
        retrieved_ids = get_retrieved_parent_ids(results)
        relevances = [rid in item["relevant_ids"] for rid in retrieved_ids[:top_k]]

        metrics = {
            "precision@5": precision_at_k(retrieved_ids, item["relevant_ids"], 5),
            "recall@5": recall_at_k(retrieved_ids, item["relevant_ids"], 5),
            "ndcg@5": ndcg_at_k(relevances, 5),
            "mrr": reciprocal_rank(retrieved_ids, item["relevant_ids"]),
            "hit@5": 1.0 if hit_at_k(retrieved_ids, item["relevant_ids"], 5) else 0.0,
        }

        category_results.setdefault(item["description"], []).append(metrics)

    category_summary = {}
    for cat, values in category_results.items():
        category_summary[cat] = {
            k: float(np.mean([v[k] for v in values])) for k in values[0].keys()
        }
        category_summary[cat]["count"] = len(values)

    return category_summary


def analyze_distance_distribution(
    collection: Any,
    model: Any,
    ground_truth: list[dict],
    top_k: int = 10,
) -> dict[str, Any]:
    """Analyze retrieval distances for relevant vs irrelevant docs."""
    all_distances = {"relevant": [], "irrelevant": []}

    for item in ground_truth:
        results = query_vectorstore(
            item["query"],
            collection=collection,
            model=model,
            top_k=top_k,
        )

        for r in results:
            dist = r.get("distance", 1.0)
            parent_id = r.get("parent_id", "")
            if parent_id in item["relevant_ids"]:
                all_distances["relevant"].append(dist)
            else:
                all_distances["irrelevant"].append(dist)

    return {
        "relevant": {
            "mean": float(np.mean(all_distances["relevant"]))
            if all_distances["relevant"]
            else 0,
            "std": float(np.std(all_distances["relevant"]))
            if all_distances["relevant"]
            else 0,
            "min": float(np.min(all_distances["relevant"]))
            if all_distances["relevant"]
            else 0,
            "max": float(np.max(all_distances["relevant"]))
            if all_distances["relevant"]
            else 0,
        },
        "irrelevant": {
            "mean": float(np.mean(all_distances["irrelevant"]))
            if all_distances["irrelevant"]
            else 0,
            "std": float(np.std(all_distances["irrelevant"]))
            if all_distances["irrelevant"]
            else 0,
            "min": float(np.min(all_distances["irrelevant"]))
            if all_distances["irrelevant"]
            else 0,
            "max": float(np.max(all_distances["irrelevant"]))
            if all_distances["irrelevant"]
            else 0,
        },
    }


def main() -> dict[str, Any]:
    """Run evaluation and save results."""
    logger.info("Loading vectorstore and embedding model...")

    try:
        collection = load_vectorstore()
        model = get_embedding_model()
    except Exception as e:
        logger.error("Failed to load vectorstore: %s", e)
        return {"error": str(e)}

    total_chunks = collection.count()
    logger.info("Vectorstore loaded: %d chunks", total_chunks)

    logger.info("Running retrieval evaluation...")
    results = evaluate_retrieval(
        collection=collection,
        model=model,
        ground_truth=GROUND_TRUTH,
        top_k=10,
    )

    logger.info("Analyzing by category...")
    category_analysis = analyze_by_category(
        collection=collection,
        model=model,
        ground_truth=GROUND_TRUTH,
        top_k=10,
    )

    logger.info("Analyzing distance distribution...")
    distance_analysis = analyze_distance_distribution(
        collection=collection,
        model=model,
        ground_truth=GROUND_TRUTH,
        top_k=10,
    )

    full_results = {
        "evaluation_summary": results,
        "category_analysis": category_analysis,
        "distance_analysis": distance_analysis,
        "metadata": {
            "total_chunks": total_chunks,
            "num_queries": results["num_queries"],
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "vectorstore": "ChromaDB",
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(full_results, f, indent=2)

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nTotal chunks in vectorstore: {total_chunks}")
    print(f"Queries evaluated: {results['num_queries']}")

    print("\n--- Overall Metrics ---")
    summary = results["summary"]
    for metric in [
        "precision@5",
        "recall@5",
        "ndcg@5",
        "mrr",
        "map",
        "hit@5",
        "hit@1",
        "hit@3",
    ]:
        if metric in summary:
            print(
                f"{metric.upper():>15}: {summary[metric]['mean']:.4f} (±{summary[metric]['std']:.4f})"
            )

    print("\n--- Hit Rate by K ---")
    for k in [1, 3, 5, 10]:
        if f"hit@{k}" in summary:
            hit_rate = summary[f"hit@{k}"]["mean"] * 100
            print(f"Hit Rate@{k:2d}: {hit_rate:6.2f}%")

    print("\n--- Category Performance (Precision@5) ---")
    for cat, metrics in sorted(
        category_analysis.items(), key=lambda x: -x[1]["precision@5"]
    ):
        print(f"  {cat[:40]:<40}: {metrics['precision@5']:.4f} (n={metrics['count']})")

    print("\n--- Distance Analysis ---")
    print(f"  Relevant docs - Mean: {distance_analysis['relevant']['mean']:.4f}")
    print(f"  Irrelevant docs - Mean: {distance_analysis['irrelevant']['mean']:.4f}")

    print(f"\nResults saved to: {RESULTS_PATH}")
    print("=" * 60 + "\n")

    return full_results


if __name__ == "__main__":
    main()
