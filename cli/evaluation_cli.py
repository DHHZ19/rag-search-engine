import argparse

from evaluation import eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    # precision = relevant_retrieved / total_retrieved
    args = parser.parse_args()
    limit = args.limit

    results = eval(limit)

    for r in results:
        precision = r["precision"]
        print(f"k={limit}")
        print(f"- Query: {r['query']}")
        print(f"- Precision@{limit}: {precision:.4f}")
        print(f"- Recall@{limit}: {r['recall']:.4f}")
        print(f"- F1 Score: {r['f1_score']:.4f}")
        print(f"- Retrieved: {r['retrieved']}")
        print(f"- Relevant: {r['relevant']}")
        print()


if __name__ == "__main__":
    main()
