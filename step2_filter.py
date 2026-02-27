"""
Step 2: Filter and flag enriched leads before manual review.

Reads the enriched CSV from step1, applies hard exclusion rules,
and flags borderline cases for manual review.

Hard removes:
  - decisionMakerScore <= 1 (residents, students, junior staff)
  - FinalScore < 20 (too low priority to be worth contacting)
  - Institution in DO NOT CONTACT tab
  - Pharma company (keyword match + known company list)
  - Public/government hospital (state hospitals, county hospitals, VA)

Flags (lead kept but highlighted for review):
  - UNIVERSITY_HOSPITAL? — university name in institution
  - PRIVATE_NONPROFIT — private non-profit hospital, check university tab manually
  - APPR_CONFLICT — institution already being worked by a team member
  - LOW_CONFIDENCE — AI confidence was low, scores less reliable
  - APA_KEY_CONTACT — person appears in APA Key Contacts list
  - POSSIBLE_PRIVATE_PRACTICE — solo practice, probably too small

Output:
  {output}_review.xlsx  — colour-coded Excel (green = clean, yellow = flagged)
  {output}_review.csv   — same data as CSV backup
  {output}_excluded.csv — all removed records with exclusion reasons

Usage:
  python step2_filter.py --input apa2026_extra_enriched.csv --output apa2026_extra
"""

import argparse
import json
import os
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
    "pharmaceutical", "pharma", " biosciences", "biopharma", " therapeutics", "biologics",
]

PUBLIC_KNOWN = {
    "georgia regional hospital", "twin valley behavioral healthcare",
    "southwest connecticut mental health", "chicago-read mental health center",
    "austin state hospital", "metrohealth medical center",
    "metrohealth and case western", "norman regional health system",
}

PUBLIC_KEYWORDS = [
    "state hospital", "state psychiatric", "state mental health",
    "county hospital", "county mental health", "veterans affairs",
    "va medical center", " vamc",
]

PRIVATE_PRACTICE_KEYWORDS = [
    "private practice", "md llc", "md pllc", "m.d. llc", "psychiatrist.net",
    "psychiatry associates", "psychiatric associates", "associates llc",
    "associates pllc", " md ", ", md,",
]


def normalise(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower().strip())


def is_pharma(institution: str) -> bool:
    inst = institution.lower()
    return any(k in inst for k in PHARMA_KNOWN) or any(k in inst for k in PHARMA_KEYWORDS)


def is_public_hospital(institution: str, inst_type: str) -> bool:
    if inst_type not in ("hospital", "unknown"):
        return False
    inst = institution.lower()
    return any(k in inst for k in PUBLIC_KNOWN) or any(k in inst for k in PUBLIC_KEYWORDS)


def is_university_hospital(institution: str, inst_type: str) -> bool:
    """Returns True for hospitals with a university in the name — flag for manual review."""
    if inst_type != "hospital":
        return False
    return bool(re.search(r"\b(university|college of medicine|school of medicine|academic medical)\b",
                          institution.lower()))


def load_dnc(xlsx_path: str) -> set:
    """Load DO NOT CONTACT institution names from the reference spreadsheet."""
    try:
        df = pd.read_excel(xlsx_path, sheet_name="DO NOT CONTACT", header=None)
        return {str(v).strip().lower() for v in df[0].dropna()}
    except Exception as e:
        print(f"Warning: could not load DO NOT CONTACT tab: {e}")
        return set()


def load_key_contacts(xlsx_path: str) -> set:
    """Load APA Key Contacts (speakers, board members, etc.) for flagging."""
    try:
        df = pd.read_excel(xlsx_path, sheet_name="APA Key Contacts")
        cred_pattern = re.compile(
            r",?\s+(M\.?D\.?|Ph\.?D\.?|D\.?O\.?|M\.?S\.?|M\.?B\.?A\.?|'?FAPA|DFAPA).*",
            re.IGNORECASE,
        )
        names = set()
        for val in df["Name"].dropna():
            clean = cred_pattern.sub("", str(val)).strip().lower()
            if clean:
                names.add(clean)
        return names
    except Exception as e:
        print(f"Warning: could not load APA Key Contacts tab: {e}")
        return set()


def load_appr_orgs(xlsx_path: str) -> set:
    """Load organisations already being worked by the team (APPR tab)."""
    try:
        df = pd.read_excel(xlsx_path, sheet_name="APPR")
        for col in ["Organisation", "Organization", "organization", "org", "Org"]:
            if col in df.columns:
                return {str(v).strip().lower() for v in df[col].dropna()}
        return set()
    except Exception as e:
        print(f"Warning: could not load APPR tab: {e}")
        return set()


def is_dnc(institution: str, dnc_set: set) -> bool:
    inst = normalise(institution)
    return any(d in inst or inst in d for d in dnc_set)


def is_appr_conflict(institution: str, appr_orgs: set) -> bool:
    inst = normalise(institution)
    return any(org in inst or inst in org for org in appr_orgs if org and len(org) >= 4)


def is_key_contact(name: str, key_contacts: set) -> bool:
    cred_pattern = re.compile(
        r",?\s+(M\.?D\.?|Ph\.?D\.?|D\.?O\.?|M\.?S\.?|M\.?B\.?A\.?|'?FAPA|DFAPA).*",
        re.IGNORECASE,
    )
    cleaned = cred_pattern.sub("", str(name)).strip().lower()
    return cleaned in key_contacts


def main():
    parser = argparse.ArgumentParser(description="Step 2: filter and flag leads for review")
    parser.add_argument("--input",  required=True,        help="Enriched CSV from step1")
    parser.add_argument("--output", default="filtered",   help="Output filename prefix")
    parser.add_argument("--xlsx",   default=DEFAULT_XLSX, help="Path to APA reference spreadsheet")
    args = parser.parse_args()

    print(f"\nLoading {args.input}...")
    df = pd.read_csv(args.input)
    print(f"  {len(df)} records loaded")

    print(f"Loading reference data from {args.xlsx}...")
    dnc_set      = load_dnc(args.xlsx)
    appr_orgs    = load_appr_orgs(args.xlsx)
    key_contacts = load_key_contacts(args.xlsx)
    print(f"  {len(dnc_set)} DNC | {len(appr_orgs)} APPR orgs | {len(key_contacts)} key contacts")

    included = []
    excluded = []

    for _, row in df.iterrows():
        inst      = str(row.get("currentInstitution") or row.get("Institution", ""))
        inst_type = str(row.get("institutionType", "unknown")).lower()
        dm_score  = int(row.get("decisionMakerScore", 0) or 0)
        score     = float(row.get("FinalScore", 0) or 0)
        confidence = str(row.get("confidence", "low")).lower()

        # Hard removes — these leads are excluded entirely
        if dm_score <= 1:
            excluded.append({**row, "excludeReason": "RESIDENT_OR_STUDENT (DM score <= 1)"})
            continue
        if score < 20:
            excluded.append({**row, "excludeReason": f"LOW_SCORE ({score:.1f} < 20)"})
            continue
        if is_dnc(inst, dnc_set):
            excluded.append({**row, "excludeReason": "DO_NOT_CONTACT"})
            continue
        if is_pharma(inst):
            excluded.append({**row, "excludeReason": "PHARMA"})
            continue
        if is_public_hospital(inst, inst_type):
            excluded.append({**row, "excludeReason": "PUBLIC_HOSPITAL"})
            continue

        # Soft flags — lead is kept but highlighted for manual review
        flags = []

        if is_university_hospital(inst, inst_type):
            flags.append("UNIVERSITY_HOSPITAL?")
        elif inst_type == "hospital":
            flags.append("PRIVATE_NONPROFIT — check university tab")

        if is_appr_conflict(inst, appr_orgs):
            flags.append("APPR_CONFLICT — team already working this org")

        if confidence == "low":
            flags.append("LOW_CONFIDENCE — scores less reliable")

        name = str(row.get("Name", ""))
        if is_key_contact(name, key_contacts):
            flags.append("APA_KEY_CONTACT — in APA Key Contacts list")

        if any(k in inst.lower() for k in PRIVATE_PRACTICE_KEYWORDS) and inst_type == "company":
            flags.append("POSSIBLE_PRIVATE_PRACTICE")

        row_dict = row.to_dict()
        row_dict["FLAGS"] = " | ".join(flags) if flags else ""
        included.append(row_dict)

    df_out = pd.DataFrame(included)

    # Clean leads first, then flagged — within each group sorted by score descending
    df_out["_has_flag"] = df_out["FLAGS"].apply(lambda x: 0 if not x else 1)
    df_out = df_out.sort_values(["_has_flag", "FinalScore"], ascending=[True, False])
    df_out = df_out.drop(columns=["_has_flag"]).reset_index(drop=True)
    df_out.insert(0, "ReviewRank", df_out.index + 1)

    review_cols = [
        "ReviewRank", "FLAGS",
        "Name", "currentTitle", "currentInstitution", "institutionType",
        "Country", "State",
        "FinalScore", "Tier", "confidence",
        "budgetScore", "decisionMakerScore", "isDecisionMaker",
        "financialCapacity", "reasoning",
        "Appearances", "Years", "FinalPreScore",
    ]
    review_cols = [c for c in review_cols if c in df_out.columns]
    df_review = df_out[review_cols].copy()

    csv_path  = f"{args.output}_review.csv"
    excl_path = f"{args.output}_excluded.csv"
    df_review.to_csv(csv_path, index=False)
    pd.DataFrame(excluded).to_csv(excl_path, index=False)

    # Build colour-coded Excel workbook
    xlsx_path = f"{args.output}_review.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_review.to_excel(writer, sheet_name="Review Leads", index=False)
        ws = writer.sheets["Review Leads"]

        from openpyxl.styles import PatternFill, Font, Alignment
        from openpyxl.utils import get_column_letter

        # Header styling
        hdr_fill = PatternFill("solid", fgColor="1F4E79")
        hdr_font = Font(bold=True, color="FFFFFF", size=10)
        for cell in ws[1]:
            cell.fill = hdr_fill
            cell.font = hdr_font
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.row_dimensions[1].height = 30

        # Alternating row colours: green for clean leads, yellow for flagged
        green_fill  = PatternFill("solid", fgColor="E2EFDA")
        yellow_fill = PatternFill("solid", fgColor="FFF2CC")
        alt_green   = PatternFill("solid", fgColor="C6EFCE")
        alt_yellow  = PatternFill("solid", fgColor="FFEB9C")

        green_count = yellow_count = 0
        for row_idx, row_data in enumerate(df_review.itertuples(), start=2):
            is_flagged = bool(str(getattr(row_data, "FLAGS", "") or "").strip())
            if is_flagged:
                fill = alt_yellow if yellow_count % 2 else yellow_fill
                yellow_count += 1
            else:
                fill = alt_green if green_count % 2 else green_fill
                green_count += 1
            for col_idx in range(1, ws.max_column + 1):
                ws.cell(row=row_idx, column=col_idx).fill = fill

        # Column widths
        col_widths = {
            "ReviewRank": 8, "FLAGS": 45, "Name": 25, "currentTitle": 35,
            "currentInstitution": 35, "institutionType": 14, "Country": 12,
            "State": 7, "FinalScore": 10, "Tier": 6, "confidence": 10,
            "budgetScore": 8, "decisionMakerScore": 8, "isDecisionMaker": 10,
            "financialCapacity": 40, "reasoning": 50,
            "Appearances": 10, "Years": 20, "FinalPreScore": 10,
        }
        for col_idx, col_name in enumerate(review_cols, start=1):
            ws.column_dimensions[get_column_letter(col_idx)].width = col_widths.get(col_name, 15)

        ws.freeze_panes = "A2"

        # Summary sheet
        ws2 = writer.book.create_sheet("Summary")
        clean_count   = len(df_review[df_review["FLAGS"] == ""])
        flagged_count = len(df_review[df_review["FLAGS"] != ""])

        excl_reasons = {}
        for e in excluded:
            r = e.get("excludeReason", "Unknown")
            excl_reasons[r] = excl_reasons.get(r, 0) + 1

        summary_rows = [
            ["STEP 2 FILTER SUMMARY", ""],
            ["", ""],
            ["INCLUDED IN REVIEW", ""],
            ["  Clean leads", clean_count],
            ["  Flagged leads (review needed)", flagged_count],
            ["  Total for review", clean_count + flagged_count],
            ["", ""],
            ["EXCLUDED", ""],
        ]
        for reason, count in sorted(excl_reasons.items(), key=lambda x: -x[1]):
            summary_rows.append([f"  {reason}", count])
        summary_rows.append(["  Total excluded", len(excluded)])
        summary_rows.append(["", ""])
        summary_rows.append(["Next step:", f"python step3_prepare.py --input {csv_path} --output {args.output}"])

        for r_idx, row in enumerate(summary_rows, start=1):
            for c_idx, val in enumerate(row, start=1):
                cell = ws2.cell(row=r_idx, column=c_idx, value=val)
                if c_idx == 1 and str(val).isupper():
                    cell.font = Font(bold=True, color="1F4E79")
        ws2.column_dimensions["A"].width = 40
        ws2.column_dimensions["B"].width = 60

    # Print summary
    excl_by_type = {}
    for e in excluded:
        key = e.get("excludeReason", "Unknown").split(" (")[0].split("_")[0]
        excl_by_type[key] = excl_by_type.get(key, 0) + 1

    print(f"\nFilter complete")
    print(f"  Input:    {len(df)}")
    print(f"  Excluded: {len(excluded)}")
    for reason, count in sorted(excl_by_type.items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count}")
    print(f"  For review (clean):   {clean_count}")
    print(f"  For review (flagged): {flagged_count}")
    print(f"\nOpen {xlsx_path} — green = clean, yellow = needs review")
    print(f"Delete rows you don't want, save as CSV, then run:")
    print(f"  python step3_prepare.py --input {csv_path} --output {args.output}")


if __name__ == "__main__":
    main()
