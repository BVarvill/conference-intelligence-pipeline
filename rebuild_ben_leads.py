"""
rebuild_ben_leads.py
====================
Rebuilds ben_leads_ready.csv with corrected columns and proper filters.

Filters applied (all three cause a lead to be EXCLUDED):
  1. Pharma companies (Boehringer Ingelheim, J&J, Neurocrine, Otsuka, etc.)
  2. Public / government hospitals (state hospitals, county hospitals, VA)
  3. University-affiliated hospitals (checked against APA 2026.xlsx tab)
     → Clear name match: removed automatically
     → Borderline match: flagged in ben_leads_excluded.csv for manual review

Column fixes (generated fresh by Groq — no Serper needed):
  - Focus Summary          : 2-4 word specialty descriptor  (used in email body)
  - Explanation            : Why the INSTITUTION has financial capacity
  - Reason/Value/Return    : What the INSTITUTION gains from an APA TV doc

Output:
  ben_leads_ready_v2.csv      ← clean leads ready to paste into Ben Leads tab
  ben_leads_excluded.csv      ← everything removed + reason (for review)
"""

import json, os, re, time
import pandas as pd
from groq import Groq
from datetime import date

# ── Keys ───────────────────────────────────────────────────────────────────────
GROQ_KEY    = os.environ.get("GROQ_API_KEY",   "")
groq_client = Groq(api_key=GROQ_KEY)
# Primary model — best quality; falls back to FALLBACK_MODEL on TPD quota exhaustion
GROQ_MODEL       = "llama-3.3-70b-versatile"
FALLBACK_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"  # separate quota
BASE             = "/Users/benvarvill/Downloads/MRA Media work"
_current_model   = GROQ_MODEL   # tracks which model we're currently using


# ═══════════════════════════════════════════════════════════════════════════════
# FILTER 1: Pharma companies
# ═══════════════════════════════════════════════════════════════════════════════

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

def is_pharma(institution: str) -> bool:
    il = institution.lower()
    for k in PHARMA_KNOWN:
        if k in il:
            return True
    for k in PHARMA_KEYWORDS:
        if k in il:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# FILTER 2: Public / government hospitals
# ═══════════════════════════════════════════════════════════════════════════════

PUBLIC_KNOWN = {
    "georgia regional hospital",
    "twin valley behavioral healthcare",
    "southwest connecticut mental health",
    "chicago-read mental health center",
    "chicago read mental health center",
    "austin state hospital",
    "metrohealth medical center",
    "metrohealth and case western",
    "norman regional health system",
}

PUBLIC_KEYWORDS = [
    "state hospital",
    "state psychiatric",
    "state mental health",
    "county hospital",
    "county mental health",
    "veterans affairs",
    "va medical center",
    " vamc",
]

def is_public_hospital(institution: str, inst_type: str) -> bool:
    if inst_type != "hospital":
        return False
    il = institution.lower()
    for k in PUBLIC_KNOWN:
        if k in il:
            return True
    for k in PUBLIC_KEYWORDS:
        if k in il:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# FILTER 3: University-affiliated hospitals (APA 2026.xlsx tab)
# ═══════════════════════════════════════════════════════════════════════════════

# Words too common to use for matching
_GENERIC = {
    "the","and","of","at","for","in","on","to","a","an",
    "health","medical","center","hospital","clinic","care",
    "behavioral","mental","institute","services","system",
    "systems","general","community","national","american",
    "association","foundation","department","division","saint","st",
    "regional","medicine","school","college",
}

def _sig_words(s: str) -> set:
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return {w for w in s.split() if len(w) > 4 and w not in _GENERIC}

def load_uni_hospitals(xlsx_path: str) -> tuple[list, list]:
    """
    Parse the 'University Hospital & Medical S' tab.
    Returns (raw_names_list, sig_words_list).
    """
    df = pd.read_excel(xlsx_path, sheet_name="University Hospital & Medical S", header=None)
    hospitals = []
    for _, row in df.iterrows():
        cell = str(row[2]) if pd.notna(row[2]) else ""
        for line in cell.split("\n"):
            name = line.strip().lstrip("- ").strip()
            if name and name not in ("AFFILIATED HOSPITAL",) and len(name) > 4:
                hospitals.append(name)
    sigs = [_sig_words(h) for h in hospitals]
    return hospitals, sigs


def check_university_hospital(
    institution: str, inst_type: str,
    uni_hospitals: list, uni_hosp_sigs: list
) -> tuple[bool, str]:
    """
    Returns (is_excluded, reason_string).
    Three-stage check:
      Stage 1 — University/academic keywords in the institution name itself
      Stage 2 — Clear substring match against tab (min 8 chars)
      Stage 3 — Significant word overlap (≥2 unique words, each >4 chars)
    """
    if inst_type != "hospital":
        return False, ""

    il = institution.lower()

    # Stage 1: university keyword in name → confident exclusion
    if re.search(
        r"\b(university|school of medicine|college of medicine|academic medical)\b", il
    ):
        return True, "UNIVERSITY_IN_NAME"

    inst_sig = _sig_words(institution)

    for hosp, h_sig in zip(uni_hospitals, uni_hosp_sigs):
        hl = hosp.lower()

        # Stage 2: substring match (require the substring to be meaningful length)
        short, long_ = (il, hl) if len(il) <= len(hl) else (hl, il)
        if len(short) >= 8 and short in long_:
            return True, f"TAB_MATCH: {hosp}"

        # Stage 3: 2+ significant word overlap
        if len(h_sig) >= 2:  # only try if the tab entry has enough sig words
            overlap = inst_sig & h_sig
            if len(overlap) >= 2:
                return True, f"TAB_WORD_MATCH ({', '.join(sorted(overlap))}): {hosp}"

    return False, ""


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: Type classification
# ═══════════════════════════════════════════════════════════════════════════════

def classify_type(budget_score: int) -> str:
    if budget_score >= 4:
        return "(LP) Large Private"
    if budget_score >= 2:
        return "(MP) Medium Private"
    return "(SP) Small Private"


# ═══════════════════════════════════════════════════════════════════════════════
# GROQ: Generate Focus Summary, Explanation, Reason/Value columns
# ═══════════════════════════════════════════════════════════════════════════════

def generate_fields(leads_ctx: list, batch_size: int = 5) -> list:
    """
    For each lead dict (name, title, institution, inst_type, financial_capacity,
    reasoning), ask Groq to generate three fields.
    Returns a list of dicts in the same order.
    """
    all_results = []

    for i in range(0, len(leads_ctx), batch_size):
        batch = leads_ctx[i : i + batch_size]

        people_lines = []
        for j, ctx in enumerate(batch, 1):
            people_lines.append(
                f"Person {j}:\n"
                f"Name: {ctx['name']}\n"
                f"Title: {ctx['title']}\n"
                f"Institution: {ctx['institution']}\n"
                f"Institution type: {ctx['inst_type']}\n"
                f"Financial capacity (from prior research): {ctx['financial_capacity']}\n"
                f"Research notes: {ctx['reasoning']}"
            )

        prompt = (
            "You are filling in a sales lead tracking spreadsheet for Websedge, "
            "which produces $27,500 APA TV documentary features showcased at the "
            "American Psychiatric Association Annual Meeting (25,000+ attendees).\n\n"
            "For each person below, generate three fields:\n\n"
            "1. focusSummary\n"
            "   A 2-4 word phrase describing the INSTITUTION's specialty area.\n"
            "   Used in the email sentence: \"at the cutting-edge of [focusSummary]\"\n"
            "   ✓ Good: \"addiction psychiatry\", \"telepsychiatry\", "
            "\"child and adolescent psychiatry\", \"TMS therapy\", "
            "\"correctional mental health\", \"digital psychiatry\", "
            "\"integrated behavioral health\", \"psychedelic-assisted therapy\"\n"
            "   ✗ Bad: repeating the institution name, generic phrases like "
            "\"mental health services\"\n\n"
            "2. institutionExplanation\n"
            "   1-2 sentences explaining WHY the INSTITUTION (not the person) "
            "has financial resources.\n"
            "   Use the financial capacity note as your source.\n"
            "   ✓ Good: \"Large non-profit Catholic health system with $8B+ "
            "annual revenue operating across 23 states.\"\n"
            "   ✗ Bad: describing the person's job title or authority\n\n"
            "3. videoValue\n"
            "   1-2 sentences on what VALUE the INSTITUTION (not the person) "
            "gains from an APA TV documentary.\n"
            "   Think: recruitment of residents/fellows, brand visibility, "
            "showcasing research/innovation, attracting referrals or funding.\n"
            "   ✓ Good: \"Boosts psychiatry residency recruitment nationally and "
            "showcases their integrated care model to 25,000+ APA attendees.\"\n"
            "   ✗ Bad: describing what the individual person gains\n\n"
            f"People:\n\n" + "\n\n".join(people_lines) + "\n\n"
            f"Return a JSON array of exactly {len(batch)} objects. "
            "Each must have keys: focusSummary, institutionExplanation, videoValue.\n"
            "No markdown, no prose, no code fences — raw JSON array only."
        )

        global _current_model
        max_attempts = 8   # more retries to ride out rolling-window resets
        attempt = 0
        while attempt < max_attempts:
            try:
                resp = groq_client.chat.completions.create(
                    model=_current_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You write precise, specific content for sales lead "
                                "tracking spreadsheets. Return only valid JSON arrays."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=2500,
                )
                text = resp.choices[0].message.content.strip()
                # Strip any markdown fences
                text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
                # Find first JSON array
                m = re.search(r"(\[.*\])", text, re.DOTALL)
                if m:
                    text = m.group(1)
                parsed = json.loads(text)
                if len(parsed) != len(batch):
                    raise ValueError(f"Expected {len(batch)} results, got {len(parsed)}")
                all_results.extend(parsed)
                summaries = [r.get("focusSummary", "?") for r in parsed]
                print(f"  ✓ Batch {i // batch_size + 1}: {summaries}")
                break
            except Exception as e:
                err_str = str(e)
                attempt += 1
                # Auto-switch to fallback model on TPD exhaustion
                if "tokens per day" in err_str.lower() and _current_model != FALLBACK_MODEL:
                    _current_model = FALLBACK_MODEL
                    print(f"  🔄 TPD quota hit — switching to fallback model: {FALLBACK_MODEL}")
                    time.sleep(2)
                    continue
                # Parse the "Please try again in Xs" hint from Groq 429 responses
                wait_match = re.search(r"try again in (\d+)m(\d+)", err_str)
                if wait_match:
                    wait_secs = int(wait_match.group(1)) * 60 + float(wait_match.group(2)) + 5
                else:
                    wait_secs = 30 * attempt  # escalating back-off otherwise
                if attempt < max_attempts:
                    print(f"  ⏳ Rate limit — waiting {wait_secs:.0f}s then retrying "
                          f"(attempt {attempt}/{max_attempts}) [{_current_model}]...")
                    time.sleep(wait_secs)
                else:
                    print(f"  ✗ All retries exhausted for batch {i // batch_size + 1}.")
                    for _ in batch:
                        all_results.append({
                            "focusSummary": "mental health innovation",
                            "institutionExplanation": "Financial details not available.",
                            "videoValue": (
                                "APA TV documentary provides national visibility "
                                "to 25,000+ psychiatrists and mental health professionals."
                            ),
                        })

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n📂 Loading data...")

    # 1. Load lead candidates
    with open(os.path.join(BASE, "outreach_clear.json")) as f:
        all_leads = json.load(f)

    # 2. Load enriched data for financialCapacity (from the full 1,999-record run)
    enriched_df = pd.read_csv(os.path.join(BASE, "enriched_all.csv"))
    fin_cap_lookup = {
        str(row["Name"]).strip().lower(): str(row.get("financialCapacity", ""))
        for _, row in enriched_df.iterrows()
    }

    # 3. Load hooks from emails_draft.csv
    hooks_lookup = {}
    emails_csv = os.path.join(BASE, "emails_draft.csv")
    if os.path.exists(emails_csv):
        edf = pd.read_csv(emails_csv)
        for _, row in edf.iterrows():
            hooks_lookup[str(row.get("name", "")).strip().lower()] = str(row.get("hook", ""))

    # 4. Load university hospital data
    print("📋 Loading university hospital tab from APA 2026.xlsx...")
    xlsx_path = os.path.join(BASE, "APA 2026.xlsx")
    uni_hospitals, uni_hosp_sigs = load_uni_hospitals(xlsx_path)
    print(f"   {len(uni_hospitals)} university hospital entries loaded")

    # 5. Filter private practice first (same as before)
    leads = [
        l for l in all_leads
        if "private practice" not in l["institution"].lower()
    ]
    print(f"\n📊 Leads after removing private practices: {len(leads)}")

    # ── Apply all three filters ────────────────────────────────────────────────
    included = []
    excluded = []

    for lead in leads:
        inst      = lead["institution"]
        inst_type = lead.get("institutionType", "company")

        # Filter 1: Pharma
        if is_pharma(inst):
            excluded.append({**lead, "excludeReason": "PHARMA"})
            continue

        # Filter 2: Public hospital
        if is_public_hospital(inst, inst_type):
            excluded.append({**lead, "excludeReason": "PUBLIC_HOSPITAL"})
            continue

        # Filter 3: University hospital
        is_uni, uni_reason = check_university_hospital(
            inst, inst_type, uni_hospitals, uni_hosp_sigs
        )
        if is_uni:
            excluded.append({**lead, "excludeReason": f"UNIVERSITY_HOSPITAL ({uni_reason})"})
            continue

        included.append(lead)

    # Print filter summary
    pharma_x  = [e for e in excluded if e["excludeReason"] == "PHARMA"]
    public_x  = [e for e in excluded if e["excludeReason"] == "PUBLIC_HOSPITAL"]
    uni_x     = [e for e in excluded if "UNIVERSITY_HOSPITAL" in e["excludeReason"]]

    print(f"\n{'─'*60}")
    print(f"  Pharma excluded    : {len(pharma_x)}")
    print(f"  Public hospitals   : {len(public_x)}")
    print(f"  University hospitals: {len(uni_x)}")
    print(f"  → INCLUDED         : {len(included)}")
    print(f"{'─'*60}")

    print("\nAll excluded leads:")
    for e in sorted(excluded, key=lambda x: x["excludeReason"]):
        print(f"  {e['fullName']:<30} | {e['institution'][:45]:<45} | {e['excludeReason']}")

    # ── Generate Groq columns ─────────────────────────────────────────────────
    print(f"\n🤖 Generating Groq fields for {len(included)} leads...")

    contexts = []
    for l in included:
        fin_cap = fin_cap_lookup.get(l["fullName"].strip().lower(), "")
        if not fin_cap or fin_cap in ("nan", "None", ""):
            fin_cap = l.get("reasoning", "No financial data available")
        contexts.append({
            "name":               l["fullName"],
            "title":              l["title"],
            "institution":        l["institution"],
            "inst_type":          l.get("institutionType", "company"),
            "financial_capacity": fin_cap,
            "reasoning":          l.get("reasoning", ""),
        })

    enrichment = generate_fields(contexts, batch_size=5)

    # ── Build output DataFrame ────────────────────────────────────────────────
    print("\n📝 Building output spreadsheet...")
    today = date.today().strftime("%d/%m/%Y")
    rows  = []

    for lead, enr in zip(included, enrichment):
        name_key   = lead["fullName"].strip().lower()
        hook       = hooks_lookup.get(name_key, "")
        budget     = lead.get("budgetScore", 0)
        decision   = lead.get("decisionMakerScore", 0)
        size_money = min(budget * 2, 10)      # 0-5 → 0-10
        affinity   = min(decision * 2, 10)    # 0-5 → 0-10

        rows.append({
            "Date":                           today,
            "Country":                        lead.get("country", ""),
            "State":                          lead.get("state", ""),
            "Institution":                    lead["institution"],
            "Center Name":                    lead["institution"],
            "Type":                           classify_type(budget),
            "Position of Lead":               lead.get("title", ""),
            "Contact Name":                   lead["fullName"],
            "Email":                          "",
            "Tracker Theme":                  "APA 2026",
            "Focus":                          "Mental Health",
            "Focus Summary":                  enr.get("focusSummary", "mental health"),
            "Link":                           "",
            "Sent":                           "",
            "Area":                           lead.get("state", ""),
            "Result":                         "",
            "Pitch Rating":                   "",
            "Source":                         "APA 2026 Attendee List",
            "KWS Term":                       "",
            "Link to Source":                 "",
            "Attended?":                      lead.get("years", ""),
            "Affinity/10":                    affinity,
            "Affinity Explanation":           lead.get("reasoning", ""),
            "Size-Money/10":                  size_money,
            "Explanation":                    enr.get("institutionExplanation", ""),
            "Contacted? (Year/s)":            "",
            "Contact/10":                     "",
            "Explanation (Contact)":          "",
            "Potential Lead Overlap?":        "",
            "Reason/Value/Return from video": enr.get("videoValue", ""),
            "Decision Maker?":                "Yes" if decision >= 4 else "Maybe",
            "Additional Notes":               f"Tier {lead.get('tier','')} | Score {lead.get('finalScore','')} | Confidence: {lead.get('confidence','')}",
            "Autocheck Fellows":              "",
            "Paste in CRM":                   hook,
        })

    df_out = pd.DataFrame(rows)

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_path  = os.path.join(BASE, "ben_leads_ready_v2.csv")
    excl_path = os.path.join(BASE, "ben_leads_excluded.csv")

    df_out.to_csv(out_path, index=False)

    excl_cols = ["fullName", "institution", "title", "institutionType",
                 "tier", "finalScore", "confidence", "excludeReason"]
    pd.DataFrame(excluded)[excl_cols].to_csv(excl_path, index=False)

    print(f"\n{'═'*60}")
    print(f"  ✅  Included CSV  : ben_leads_ready_v2.csv  ({len(rows)} rows)")
    print(f"  📋  Excluded CSV  : ben_leads_excluded.csv  ({len(excluded)} rows)")
    print(f"{'═'*60}")

    # Sample preview
    print("\nSample Focus Summaries:")
    for _, r in df_out[["Contact Name", "Institution", "Focus Summary"]].head(8).iterrows():
        print(f"  {r['Contact Name']:<30} | {r['Institution'][:40]:<40} | {r['Focus Summary']}")

    print("\nSample Explanations (Size-Money column):")
    for _, r in df_out[["Contact Name", "Explanation"]].head(5).iterrows():
        print(f"  {r['Contact Name']:<30} | {str(r['Explanation'])[:80]}")


if __name__ == "__main__":
    main()
