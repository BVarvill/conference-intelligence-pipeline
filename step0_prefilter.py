"""
Step 0: Pre-filter raw attendee data before AI enrichment.

Removes obvious non-leads using instant keyword/list checks — no API calls needed.
Run this first to reduce the number of records sent to step1, saving Groq and Serper quota.

Hard removes:
  - Pharma companies (keyword match + known company list)
  - DO NOT CONTACT institutions (from APA 20XX.xlsx reference sheet)
  - Non-US leads (override with --keep-all-countries if needed)

Soft flags (kept, noted for step2 to confirm):
  - POSSIBLE_PUBLIC — state hospitals, county hospitals, VA centres

Usage:
  python step0_prefilter.py --input Extra2000Leads.csv --output apa2026_extra
  python step0_prefilter.py --input Extra2000Leads.csv --output test400 --limit 400

Output feeds directly into step1:
  python step1_enrich.py --input {output}_step1_input.csv --output {output}
"""

import argparse
import re
import pandas as pd


DEFAULT_XLSX = "APA 2026.xlsx"

PHARMA_KNOWN = {
    "boehringer ingelheim", "bristol myers squibb", "bristol-myers squibb",
    "janssen", "johnson & johnson", "johnson and johnson",
    "intra-cellular therapies", "neurocrine biosciences", "neurocrine",
    "otsuka pharmaceutical", "otsuka", "sage therapeutics",
    "takeda", "biogen", "indivior", "alkermes",
    "abbvie", "pfizer", "eli lilly", "astrazeneca", "novartis",
    "roche", "merck", "lundbeck", "sunovion", "acadia", "axsome",
}

PHARMA_KEYWORDS = [
    "pharmaceutical", "pharma", " biosciences", "biopharma",
    " therapeutics", "biologics",
]

PUBLIC_KEYWORDS = [
    "state hospital", "state psychiatric", "state mental health",
    "county hospital", "county mental health", "veterans affairs",
    "va medical center", " vamc",
]

US_VALUES = {"usa", "united states", "us"}


def normalise(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower().strip())


def is_pharma(institution: str) -> bool:
    inst = normalise(institution)
    return any(k in inst for k in PHARMA_KNOWN) or any(k in inst for k in PHARMA_KEYWORDS)


def is_possible_public(institution: str) -> bool:
    """Soft flag only — step2 applies the definitive public hospital check."""
    inst = normalise(institution)
    return any(k in inst for k in PUBLIC_KEYWORDS)


def load_dnc(xlsx_path: str) -> set:
    """Load the DO NOT CONTACT list from the reference spreadsheet."""
    try:
        df = pd.read_excel(xlsx_path, sheet_name="DO NOT CONTACT", header=None)
        return {normalise(v) for v in df[0].dropna()}
    except Exception as e:
        print(f"Warning: could not load DO NOT CONTACT tab: {e}")
        return set()


def is_dnc(institution: str, dnc_set: set) -> bool:
    inst = normalise(institution)
    return any(d in inst or inst in d for d in dnc_set if d)


def main():
    parser = argparse.ArgumentParser(description="Step 0: pre-filter attendees before AI enrichment")
    parser.add_argument("--input",               required=True,          help="Raw attendee CSV")
    parser.add_argument("--output",              default="prefiltered",  help="Output filename prefix")
    parser.add_argument("--xlsx",                default=DEFAULT_XLSX,   help="Path to APA reference spreadsheet")
    parser.add_argument("--limit",               type=int, default=None, help="Only process first N rows (for testing)")
    parser.add_argument("--keep-all-countries",  action="store_true",    help="Keep non-US leads (default: US only)")
    args = parser.parse_args()

    us_only = not args.keep_all_countries

    print(f"\nLoading {args.input}...")
    df = pd.read_csv(args.input)

    if args.limit:
        df = df.head(args.limit)
        print(f"  Test mode: {len(df)} rows")
    else:
        print(f"  {len(df)} total records")

    print(f"Loading reference data from {args.xlsx}...")
    dnc_set = load_dnc(args.xlsx)
    print(f"  {len(dnc_set)} DO NOT CONTACT entries")
    if us_only:
        print("  Non-US leads will be removed (pass --keep-all-countries to override)")

    kept = []
    removed = []

    for _, row in df.iterrows():
        institution = str(row.get("Institution", "") or "")
        country = str(row.get("Country", "") or "").strip()
        remove_reason = None

        if us_only and normalise(country) not in US_VALUES and country != "":
            remove_reason = f"NON_US ({country})"
        elif is_pharma(institution):
            remove_reason = "PHARMA"
        elif is_dnc(institution, dnc_set):
            remove_reason = "DO_NOT_CONTACT"

        record = row.to_dict()
        if remove_reason:
            record["removeReason"] = remove_reason
            removed.append(record)
        else:
            if is_possible_public(institution):
                record["_preFlag"] = "POSSIBLE_PUBLIC"
            kept.append(record)

    df_kept = pd.DataFrame(kept)
    df_removed = pd.DataFrame(removed)

    step1_path = f"{args.output}_step1_input.csv"
    removed_path = f"{args.output}_pre_removed.csv"

    df_kept.to_csv(step1_path, index=False)
    if not df_removed.empty:
        df_removed.to_csv(removed_path, index=False)

    pharma_ct = sum(1 for r in removed if "PHARMA" in r.get("removeReason", ""))
    dnc_ct = sum(1 for r in removed if "DO_NOT" in r.get("removeReason", ""))
    non_us_ct = sum(1 for r in removed if "NON_US" in r.get("removeReason", ""))
    pub_flag_ct = sum(1 for r in kept if r.get("_preFlag") == "POSSIBLE_PUBLIC")
    saved_pct = round(len(removed) / len(df) * 100) if len(df) else 0

    print(f"\nPre-filter complete")
    print(f"  Input:          {len(df)}")
    print(f"  Removed:        {len(removed)} ({saved_pct}% of API calls saved)")
    print(f"    Pharma:       {pharma_ct}")
    print(f"    DNC:          {dnc_ct}")
    print(f"    Non-US:       {non_us_ct}")
    print(f"  Kept for step1: {len(kept)} (of which {pub_flag_ct} soft-flagged as possible public)")
    print(f"\nNext: python step1_enrich.py --input {step1_path} --output {args.output}")


if __name__ == "__main__":
    main()
