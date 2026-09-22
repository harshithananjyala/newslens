"""Compare keyword, semantic and hybrid search quality.

    python -m backend.evaluate
Uses natural-language queries written WITHOUT copying article wording. A result counts as
relevant when its AG News topic label matches the query's topic (a proxy for relevance,
not human judgement). Metric: precision@10. Writes benchmarks/quality.md.
"""
from . import config
from .engine import SearchEngine

QUERIES = {
    "World": [
        "government leaders negotiate peace deal", "election results and political unrest abroad",
        "united nations condemns violence", "diplomats meet to discuss nuclear program",
        "soldiers clash in disputed territory", "president visits foreign country for talks",
        "refugees flee war zone", "terror attack kills civilians",
    ],
    "Sports": [
        "team wins championship final", "star player injured before season",
        "coach fired after losing streak", "athletes compete at the olympic games",
        "striker scores late goal", "quarterback signs contract extension",
        "tennis champion beats rival in straight sets", "pitcher throws no hitter",
    ],
    "Business": [
        "company profits fall short of expectations", "stock market drops as investors worry",
        "oil prices climb to new high", "bank announces job cuts",
        "retailer reports quarterly earnings", "merger deal between two large firms",
        "central bank raises interest rates", "airline loses money amid rising fuel costs",
    ],
    "Sci/Tech": [
        "new software vulnerability discovered", "scientists discover distant planet",
        "smartphone maker unveils new device", "researchers study climate change effects",
        "internet company launches search feature", "space probe sends images back to earth",
        "computer chip maker announces faster processor", "biologists sequence genome",
    ],
}


def main(k: int = 10):
    engine = SearchEngine()
    totals = {m: [] for m in ("keyword", "semantic", "hybrid")}
    per_topic = {t: {m: [] for m in totals} for t in QUERIES}
    for topic, qs in QUERIES.items():
        for q in qs:
            for mode in totals:
                res = engine.search(q, mode, k)["results"]
                p = sum(r["category"] == topic for r in res) / k
                totals[mode].append(p)
                per_topic[topic][mode].append(p)

    avg = lambda xs: sum(xs) / len(xs)
    table = "| Topic | Keyword | Semantic | Hybrid |\n|---|---|---|---|\n"
    for t in QUERIES:
        table += f"| {t} | " + " | ".join(f"{avg(per_topic[t][m]):.2f}" for m in totals) + " |\n"
    table += "| **Overall** | " + " | ".join(f"**{avg(totals[m]):.2f}**" for m in totals) + " |\n"
    n = sum(len(v) for v in QUERIES.values())
    text = (f"# Search quality (topic precision@{k})\n\n{n} natural-language queries, "
            f"{len(engine.texts):,} documents.\n\n{table}")
    print("\n" + text)
    config.BENCH_DIR.mkdir(exist_ok=True)
    (config.BENCH_DIR / "quality.md").write_text(text)
    print("Saved to benchmarks/quality.md")


if __name__ == "__main__":
    main()
