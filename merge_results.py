"""
Merge multiple daily enriched CSV files into a single ranked output.

Expects files matching the pattern enriched_day*.csv in the current directory.
Deduplicates by name (keeps the first/highest-scoring occurrence), re-ranks globally,
and assigns A/B/C/D tiers.

Usage:
  python merge_results.py
"""

import glob
import pandas as pd


INPUT_PATTERN = "enriched_day*.csv"
OUTPUT_FILE   = "enriched_all.csv"


def assign_tier(score: float) -> str:
    if score >= 65: return "A"
    if score >= 45: return "B"
    if score >= 25: return "C"
    return "D"


def main():
    files = sorted(glob.glob(INPUT_PATTERN))
    if not files:
        print(f"No files found matching '{INPUT_PATTERN}'. Run the daily enrichment first.")
        return

    print(f"Found {len(files)} file(s): {', '.join(files)}")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        frames.append(df)
        print(f"  {f}: {len(df)} records")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Total before dedup: {len(combined)}")

    # Keep first occurrence per name (highest-scoring day's result comes first)
    combined = combined.drop_duplicates(subset=["Name"], keep="first")
    print(f"  Total after dedup:  {len(combined)}")

    combined = combined.sort_values("FinalScore", ascending=False).reset_index(drop=True)
    combined["Tier"] = combined["FinalScore"].apply(assign_tier)
    combined["Rank"] = combined.index + 1

    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")

    tiers = combined["Tier"].value_counts().reindex(["A", "B", "C", "D"], fill_value=0)
    print(f"\n  Total: {len(combined)}")
    print(f"  Tier A (email first):   {tiers['A']}")
    print(f"  Tier B (worth contact): {tiers['B']}")
    print(f"  Tier C (possible):      {tiers['C']}")
    print(f"  Tier D (low priority):  {tiers['D']}")

    cols = ["Rank", "Tier", "Name", "currentTitle", "currentInstitution",
            "budgetScore", "decisionMakerScore", "FinalScore"]
    available = [c for c in cols if c in combined.columns]
    print(f"\nTop 15:")
    print(combined[available].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
