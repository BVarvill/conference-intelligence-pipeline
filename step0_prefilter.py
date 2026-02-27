"""
step0_prefilter.py
==================
Websedge Conference Lead Pipeline — Step 0: Pre-filter (before AI enrichment)
-------------------------------------------------------------------------------
Removes obvious non-leads BEFORE step1 runs, saving Groq + Serper API calls.
Zero API calls — all filtering is instant keyword/list-based logic.

Hard removes (never reach step1):
  - Pharma companies     (name keywords + known company list)
  - DNC institutions     (DO NOT CONTACT tab in APA 20XX.xlsx)
  - Non-US leads         (Ben focuses on US private — override with --keep-all-countries)

Soft flags only (kept, noted in output for step2 to confirm):
  - POSSIBLE_PUBLIC      (state hospital / county / VA keywords in name)

Use --limit N to test on the first N rows before running the full batch.

Usage:
  # Full run
  python step0_prefilter.py --input Extra2000Leads.csv --output apa2026_extra

  # Test on first 400 leads
  python step0_prefilter.py --input Extra2000Leads.csv --output test400 --limit 400

  # Keep non-US leads too
  python step0_prefilter.py --input Extra2000Leads.csv --output apa2026_extra --keep-all-countries

Next step after this:
  python step1_enrich.py --input {output}_step1_input.csv --output {output}
"""

import argparse, re
import pandas as pd

DEFAULT_XLSX = "/Users/benvarvill/Downloads/MRA Media work/APA 2026.xlsx"

# ── Filter lists (same source of truth as step2) ─────────────────────────────

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower().strip())


def is_pharma(inst: str) -> bool:
    il = _norm(inst)
    return any(k in il for k in PHARMA_KNOWN) or any(k in il for k in PHARMA_KEYWORDS)


def is_possible_public(inst: str) -> bool:
    """Soft flag — step2 does definitive check once institutionType is known."""
    il = _norm(inst)
    return any(k in il for k in PUBLIC_KEYWORDS)


def load_dnc(xlsx_path: str) -> set:
    try:
        df = pd.read_excel(xlsx_path, sheet_name="DO NOT CONTACT", header=None)
        return {_norm(v) for v in df[0].dropna()}
    except Exception as e:
        print(f"  ⚠  Could not load DO NOT CONTACT tab: {e}")
        return set()


def is_dnc(inst: str, dnc_set: set) -> bool:
    il = _norm(inst)
    return any(d in il or il in d for d in dnc_set if d)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Websedge Step 0 — Pre-filter before AI enrichment")
    parser.add_argument("--input",               required=True,          help="Raw attendee CSV")
    parser.add_argument("--output",              default="prefiltered",  help="Output filename prefix")
    parser.add_argument("--xlsx",                default=DEFAULT_XLSX,   help="Path to APA 20XX.xlsx")
    parser.add_argument("--limit",               type=int, default=None, help="Only use first N rows (test mode)")
    parser.add_argument("--keep-all-countries",  action="store_true",    help="Keep non-US leads (default: non-US removed)")
    args = parser.parse_args()

    us_only = not args.keep_all_countries

    print(f"\n📂 Loading {args.input}...")
    df = pd.read_csv(args.input)

    if args.limit:
        df = df.head(args.limit)
        print(f"   ⚡ Test mode: using first {len(df)} rows (of {args.limit} requested)")
    else:
        print(f"   {len(df)} total records")

    print(f"\n📋 Loading reference data from {args.xlsx}...")
    dnc_set = load_dnc(args.xlsx)
    print(f"   {len(dnc_set)} DO NOT CONTACT entries")
    if us_only:
        print("   🌍 Non-US leads will be removed (pass --keep-all-countries to override)")

    # ── Apply filters ─────────────────────────────────────────────────────────
    print("\n🔍 Filtering...")
    kept    = []
    removed = []

    for _, row in df.iterrows():
        inst    = str(row.get("Institution", "") or "")
        country = str(row.get("Country", "") or "").strip()
        reason  = None

        if us_only and _norm(country) not in US_VALUES and country != "":
            reason = f"NON_US ({country})"
        elif is_pharma(inst):
            reason = "PHARMA"
        elif is_dnc(inst, dnc_set):
            reason = "DO_NOT_CONTACT"

        r = row.to_dict()
        if reason:
            r["removeReason"] = reason
            removed.append(r)
        else:
            if is_possible_public(inst):
                r["_preFlag"] = "POSSIBLE_PUBLIC"   # step2 will confirm
            kept.append(r)

    df_kept    = pd.DataFrame(kept)
    df_removed = pd.DataFrame(removed)

    # ── Save outputs ──────────────────────────────────────────────────────────
    step1_path   = f"{args.output}_step1_input.csv"
    removed_path = f"{args.output}_pre_removed.csv"

    df_kept.to_csv(step1_path, index=False)
    if not df_removed.empty:
        df_removed.to_csv(removed_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    pharma_ct  = sum(1 for r in removed if "PHARMA"        in r.get("removeReason",""))
    dnc_ct     = sum(1 for r in removed if "DO_NOT"        in r.get("removeReason",""))
    non_us_ct  = sum(1 for r in removed if "NON_US"        in r.get("removeReason",""))
    pub_fl_ct  = sum(1 for r in kept    if r.get("_preFlag") == "POSSIBLE_PUBLIC")
    saved_pct  = round(len(removed) / len(df) * 100) if len(df) else 0

    print(f"""
╔══════════════════════════════════════════╗
║      STEP 0 PRE-FILTER COMPLETE          ║
╠══════════════════════════════════════════╣
║  Input records    : {len(df):<20} ║
║  ✗ Removed        : {len(removed):<20} ║
║     Pharma        : {pharma_ct:<20} ║
║     DNC           : {dnc_ct:<20} ║
║     Non-US        : {non_us_ct:<20} ║
║  ~ Kept for step1 : {len(kept):<20} ║
║     Possible pub. : {pub_fl_ct:<20} ║
║  API calls saved  : ~{len(removed)} ({saved_pct}%){' ' * max(0,15-len(f'{len(removed)} ({saved_pct}%)'))} ║
╚══════════════════════════════════════════╝

▶  Next:
   python step1_enrich.py --input {step1_path} --output {args.output}
    """)


if __name__ == "__main__":
    main()
