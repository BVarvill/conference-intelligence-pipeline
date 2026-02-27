"""
merge_results.py
================
Combines the 5 daily enriched CSV files into one master ranked file.

Usage:
  python merge_results.py

Expects files named enriched_day1.csv through enriched_day5.csv in the
same folder. Outputs:
  - enriched_all.csv   (all records, sorted by FinalScore descending)
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
    print(f"\nTotal records before dedup: {len(combined)}")

    # Drop duplicates — keep the first occurrence (highest-scoring day's result)
    combined = combined.drop_duplicates(subset=["Name"], keep="first")
    print(f"Total records after dedup:  {len(combined)}")

    # Re-sort and re-rank globally
    combined = combined.sort_values("FinalScore", ascending=False).reset_index(drop=True)
    combined["Tier"] = combined["FinalScore"].apply(assign_tier)
    combined["Rank"] = combined.index + 1

    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved → {OUTPUT_FILE}")

    # Summary
    tiers = combined["Tier"].value_counts().reindex(["A","B","C","D"], fill_value=0)
    print(f"""
╔══════════════════════════════════════════╗
║           MERGE COMPLETE                 ║
╠══════════════════════════════════════════╣
║  Total records           : {len(combined):<14} ║
║  Tier A (email first)    : {tiers['A']:<14} ║
║  Tier B (worth contacting: {tiers['B']:<14} ║
║  Tier C (possible)       : {tiers['C']:<14} ║
║  Tier D (low priority)   : {tiers['D']:<14} ║
╚══════════════════════════════════════════╝
    """)

    print("Top 15 leads:")
    cols = ["Rank", "Tier", "Name", "currentTitle", "currentInstitution", "budgetScore", "decisionMakerScore", "FinalScore"]
    available = [c for c in cols if c in combined.columns]
    print(combined[available].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
