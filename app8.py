# app8.py  (app7 + "Zero JIRA Time" roster feature)
# NEW IN app8:
#  - Optional roster upload (CSV/XLSX with Name + Team columns) via sidebar
#  - After processing JIRA file, compares roster vs. who actually logged time
#  - Displays a "⚠️ No JIRA Time Logged This Week" table grouped by team
#  - Includes zero-logger section in Confluence publish
#  - Draws a "Missing from JIRA" chart showing 0h bars so the gaps are visible
#  - All app7 functionality preserved unchanged

import io
import os
import re
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import streamlit as st

# ------------- Streamlit page config -------------
st.set_page_config(page_title="Time Utilization Analyzer → Publish to Confluence", layout="wide")

# Hard-default space key for the HTS I&O space (adjust if needed)
CONFLUENCE_SPACE_KEY = "HTSIO"

# Weekly capacity per person (manager request)
WEEKLY_CAPACITY_HOURS = 35.0


def _secret(key: str, default: str = "") -> str:
    """
    Read a secret from st.secrets first (covers both .streamlit/secrets.toml
    locally and Streamlit Cloud's Secrets UI), then fall back to os.getenv.
    This is why your credentials pre-fill when the secrets file is present.
    """
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError, Exception):
        return os.getenv(key, default)


# =====================================================================
#  Manager-provided mappings (Department canonicalization)
# =====================================================================

# Canonical department labels (what you want shown in Combined)
# Keys should be "normalized" forms (see _norm_key and _clean_dept_for_lookup).
DEPT_CANONICAL_MAP: Dict[str, str] = {
    # Endpoint Engineering
    "endpt": "Endpoint Engineering",
    "endpoint engineering": "Endpoint Engineering",
    "endpoint eng": "Endpoint Engineering",
    "endpoint engineer": "Endpoint Engineering",
    "hts endpoint engineering": "Endpoint Engineering",
    "hts endpoint eng": "Endpoint Engineering",

    # Network
    "network": "Network",
    "hts network": "Network",
    "hts cdw network": "Network",
    "cdw network": "Network",
    "hts cdw - network": "Network",
    "hts cdw -network": "Network",

    # Directory Services
    "directory services": "Directory Services",
    "hts directory services": "Directory Services",
    "hts dir services": "Directory Services",
    "dir services": "Directory Services",

    # SRE
    "sre": "SRE",
    "hts sre": "SRE",
    "hts sre google": "SRE",

    # Unified Communications
    "unified communications": "Unified Communications",
    "hts uc": "Unified Communications",
    "uc": "Unified Communications",

    # Identity
    "identity": "Identity",
    "hts identity": "Identity",
    "hts identity team": "Identity",
    "identity team": "Identity",

    # Collaboration & Productivity Applications
    "collaboration productivity applications": "Collaboration & Productivity Applications",
    "collaboration and productivity applications": "Collaboration & Productivity Applications",
    "collaboration productivity apps": "Collaboration & Productivity Applications",
    "collaboration and productivity apps": "Collaboration & Productivity Applications",
    "hts messaging": "Collaboration & Productivity Applications",
    "hts miro support": "Collaboration & Productivity Applications",
    "hts slack support": "Collaboration & Productivity Applications",
}


def _norm_key(s: str) -> str:
    """
    Strong normalization for mapping lookups.
    - lower
    - remove punctuation
    - normalize separators (&,/,-,_) to spaces
    - collapse whitespace
    """
    s = "" if s is None else str(s)
    s = s.strip().lower()

    # normalize common separators to spaces
    s = s.replace("&", " and ")
    s = re.sub(r"[/\\\-\_]+", " ", s)

    # drop parentheses contents but keep words outside
    s = re.sub(r"\([^)]*\)", " ", s)

    # remove any remaining non-alnum chars
    s = re.sub(r"[^a-z0-9\s]+", " ", s)

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _clean_dept_for_lookup(raw_dept: str) -> str:
    """
    Applies normalization + prefix cleanup so variants map.
    Examples:
      "HTS Endpoint Engineering" -> "endpoint engineering"
      "HTS CDW Network" -> "network"
      "HTS - Directory Services" -> "directory services"
    """
    k = _norm_key(raw_dept)

    # Remove common org prefixes that create duplicates
    # Do this AFTER normalization so it's consistent.
    # Only remove at the beginning.
    k = re.sub(r"^(hts\s+cdw\s+)", "", k)
    k = re.sub(r"^(hts\s+)", "", k)
    k = re.sub(r"^(hearst\s+)", "", k)

    k = k.strip()
    return k


def canonicalize_department(dept: str) -> str:
    if dept is None or (isinstance(dept, float) and np.isnan(dept)):
        return "Unassigned/Unknown"

    original = str(dept).strip()
    # First try cleaned key (prefix-stripped) so "HTS X" merges into "X"
    key = _clean_dept_for_lookup(original)

    # Direct hit after cleanup
    mapped = DEPT_CANONICAL_MAP.get(key)
    if mapped:
        return mapped

    # If not found, try normalized-but-not-prefix-stripped as a fallback
    key2 = _norm_key(original)
    mapped2 = DEPT_CANONICAL_MAP.get(key2)
    if mapped2:
        return mapped2

    # No match → return original label
    return original


# =====================================================================
#  Person normalization (e.g., sean.kingston -> Sean Kingston)
# =====================================================================

def canonicalize_person(p: str) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "Unassigned/Unknown"

    s = str(p).strip()

    # If it's an email, keep only local part
    if "@" in s:
        s = s.split("@", 1)[0]

    # Convert username-like patterns "first.last" -> "First Last"
    if "." in s and " " not in s:
        s = s.replace(".", " ")

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # Title case but keep common initials/abbrev decent
    s = " ".join([w.capitalize() if w.islower() else w for w in s.split(" ")])

    return s


# =====================================================================
#  Helpers: filename date parsing (for page title)
# =====================================================================

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def extract_mmdd_from_timetracking_filename(name: str) -> Optional[str]:
    """
    Extract MM/DD/YY from a ServiceNow Time Tracking filename.
    Handles two formats:

    Old (spaces/parens): 'Time Tracking for I&O teams (List report)- 11/10.xlsx'
                          -> '11/10/25'  (year from current date)

    New (underscores):   'Time_Tracking_for_I_O_teams__List_report__2_9.xlsx'
                          -> '02/09/26'  (year from current date)
    """
    cur_yy = datetime.now().strftime("%y")   # e.g. "26"

    # Old format: 'Time Tracking for I&O teams (List report)- MM/DD'
    m = re.search(
        r"Time[\s_]+Tracking[\s_]+for[\s_]+I[\s_&O]+teams[\s_]*[\(\[]?List[\s_]+report[\)\]]?\s*[-_]+\s*"
        r"([0-9]{1,2})[/_]([0-9]{1,2})",
        name,
        re.I,
    )
    if m:
        mo = m.group(1).zfill(2)
        dy = m.group(2).zfill(2)
        return f"{mo}/{dy}/{cur_yy}"

    # New underscore format: '...__List_report__<M>_<D>.xlsx'
    # Anchored to the end of the stem so we don't pick up stray digits in the middle.
    m2 = re.search(
        r"[Ll]ist_[Rr]eport__([0-9]{1,2})_([0-9]{1,2})",
        name,
        re.I,
    )
    if m2:
        mo = m2.group(1).zfill(2)
        dy = m2.group(2).zfill(2)
        return f"{mo}/{dy}/{cur_yy}"

    return None


def extract_range_from_jira_filename(name: str) -> Optional[str]:
    """
    Extract MM/DD/YY–MM/DD/YY from a Jira/Tempo report filename, e.g.:

      'Team_CCOE_02_Feb_26_08_Feb_26_Raw_Data_Logged_Time.xlsx'
       -> '02/02/26–02/08/26'

      'I&O_Report_Only_Projects_03_Nov_25_09_Nov_25.xlsx'
       -> '11/03/25–11/09/25'

    Pattern: ..._<DD>_<Mon>_<YY>_<DD>_<Mon>_<YY>...
    Year digits (e.g. "26") are now included in the output.
    """
    pattern = (
        r"(\d{1,2})_"
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)_"
        r"(\d{2})_"
        r"(\d{1,2})_"
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)_"
        r"(\d{2})"
    )
    m = re.search(pattern, name, re.IGNORECASE)
    if not m:
        return None

    d1, mon1, y1, d2, mon2, y2 = m.groups()
    mon1 = mon1.lower()
    mon2 = mon2.lower()
    if mon1 not in MONTH_MAP or mon2 not in MONTH_MAP:
        return None

    m1 = str(MONTH_MAP[mon1]).zfill(2)
    m2 = str(MONTH_MAP[mon2]).zfill(2)
    d1 = d1.zfill(2)
    d2 = d2.zfill(2)
    # y1/y2 are already 2-digit strings like "26"
    return f"{m1}/{d1}/{y1}–{m2}/{d2}/{y2}"


def pick_week_title_from_uploads(uploaded_files) -> str:
    """
    PRIORITY:
      1) If ANY uploaded file looks like a Jira/Tempo report with a date range, use that:
           'Time Utilization Report (MM/DD/YY–MM/DD/YY)'
             e.g. 'Time Utilization Report (02/02/26–02/08/26)'
      2) Else, if we find the ServiceNow-style Time Tracking file, use its date:
           'Time Utilization Report (MM/DD/YY)'
      3) Else, fall back to today's date MM/DD/YY.
    """
    date_range = None
    mmdd = None

    for f in uploaded_files or []:
        date_range = extract_range_from_jira_filename(getattr(f, "name", ""))
        if date_range:
            break

    if date_range:
        return f"Time Utilization Report ({date_range})"

    for f in uploaded_files or []:
        mmdd = extract_mmdd_from_timetracking_filename(getattr(f, "name", ""))
        if mmdd:
            break

    if not mmdd:
        mmdd = datetime.now().strftime("%m/%d/%y")   # e.g. "02/10/26"
    return f"Time Utilization Report ({mmdd})"


# =====================================================================
#  Helpers: column detection & transforms
# =====================================================================

def detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(cands: List[str]) -> Optional[str]:
        for key in cands:
            if key in cols:
                return cols[key]
            for c in df.columns:
                if key.replace(" ", "") == c.lower().replace(" ", ""):
                    return c
        return None

    work_col = pick(["work time", "logged hours", "time spent", "hours", "duration", "time worked", "work_time"])
    person_col = pick(["assigned to", "full name", "assignee", "user", "name", "resource", "owner", "person"])
    dept_col = pick(["tempo team", "assignment group", "team", "department", "group", "org", "department/team"])
    date_col = pick(["created", "date", "work date", "updated"])  # optional
    return work_col, person_col, dept_col, date_col


def detect_platform(df: pd.DataFrame) -> str:
    lc = [c.lower() for c in df.columns]
    j = {"epic", "story", "issue key", "issue id", "sprint", "project key", "logged hours", "tempo team"}
    s = {"assignment group", "assigned to", "incident", "problem", "change", "service", "work time"}
    j_score = sum(1 for x in j if x in lc)
    s_score = sum(1 for x in s if x in lc)
    if j_score > s_score:
        return "JIRA"
    if s_score > j_score:
        return "ServiceNow"
    return "ServiceNow"


def normalize_hours(series: pd.Series, platform_hint: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if platform_hint == "ServiceNow":
        s = s / 3600.0
    return s.round(2)


def build_pivot(df: pd.DataFrame, person_col: Optional[str], dept_col: Optional[str], hours_col="Hours") -> pd.DataFrame:
    data = df.copy()
    if person_col is None:
        person_col = "__Person__"
        data[person_col] = "Unassigned/Unknown"
    if dept_col is None:
        dept_col = "__Dept__"
        data[dept_col] = "Unassigned/Unknown"

    grp = data.groupby([dept_col, person_col], dropna=False)[hours_col].sum().reset_index()
    grp["__DeptTotal__"] = grp.groupby(dept_col)[hours_col].transform("sum")
    grp = grp.sort_values(["__DeptTotal__", dept_col, hours_col], ascending=[False, True, False]) \
        .drop(columns="__DeptTotal__")
    return grp.rename(columns={dept_col: "Department/Team", person_col: "Person", hours_col: "Total Hours"})


# =====================================================================
#  Canonicalization + Combined-only dedupe (person belongs to ONE dept)
# =====================================================================

def canonicalize_pivot(pivot_df: pd.DataFrame) -> pd.DataFrame:
    if pivot_df is None or pivot_df.empty:
        return pivot_df

    df = pivot_df.copy()
    df["Department/Team"] = df["Department/Team"].apply(canonicalize_department)
    df["Person"] = df["Person"].apply(canonicalize_person)
    df["Total Hours"] = pd.to_numeric(df["Total Hours"], errors="coerce").fillna(0.0)

    # Re-aggregate after canonicalization
    df = df.groupby(["Department/Team", "Person"], as_index=False)["Total Hours"].sum()
    return df


def assign_each_person_to_one_dept(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a person appears only once in the combined pivot.
    Rule: pick the department where that person has the MOST hours.
    Then sum ALL their hours into that department.
    """
    if combined_df is None or combined_df.empty:
        return combined_df

    df = combined_df.copy()
    df["Total Hours"] = pd.to_numeric(df["Total Hours"], errors="coerce").fillna(0.0)

    # total hours per person across all depts
    person_total = df.groupby("Person", as_index=False)["Total Hours"].sum().rename(
        columns={"Total Hours": "PersonTotalHours"}
    )
    df = df.merge(person_total, on="Person", how="left")

    # pick dept with max hours per person (tie-breaker: alphabetical dept)
    df_sorted = df.sort_values(["Person", "Total Hours", "Department/Team"], ascending=[True, False, True])
    winners = df_sorted.groupby("Person", as_index=False).first()[["Person", "Department/Team"]].rename(
        columns={"Department/Team": "WinnerDept"}
    )

    df = df.merge(winners, on="Person", how="left")

    # move the person's TOTAL to the winner dept, drop the split rows
    df_winner = df[["Person", "WinnerDept", "PersonTotalHours"]].drop_duplicates()
    df_winner = df_winner.rename(columns={"WinnerDept": "Department/Team", "PersonTotalHours": "Total Hours"})

    # now aggregate (safe)
    out = df_winner.groupby(["Department/Team", "Person"], as_index=False)["Total Hours"].sum()
    return out


# =====================================================================
#  NEW (app8): Roster helpers – find people who logged 0 JIRA hours
# =====================================================================

def load_roster(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Load a roster CSV or XLSX.
    Expected columns (flexible detection): Name/Full Name + Team/Department/Group.
    Returns a DataFrame with clean columns: ['Person', 'Department/Team']
    """
    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            df = pd.read_excel(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"Could not read roster file: {e}")

    cols_lower = {c.lower().strip(): c for c in df.columns}

    # Detect name column
    name_col = None
    for candidate in ["full name", "name", "person", "display name", "employee"]:
        if candidate in cols_lower:
            name_col = cols_lower[candidate]
            break
    if name_col is None:
        raise ValueError(
            "Roster file must have a column named one of: Full Name, Name, Person, Display Name, Employee.\n"
            f"Found columns: {list(df.columns)}"
        )

    # Detect team/dept column
    team_col = None
    for candidate in ["team", "department", "dept", "group", "assignment group", "tempo team", "department/team"]:
        if candidate in cols_lower:
            team_col = cols_lower[candidate]
            break
    if team_col is None:
        raise ValueError(
            "Roster file must have a column named one of: Team, Department, Dept, Group, Assignment Group, Tempo Team.\n"
            f"Found columns: {list(df.columns)}"
        )

    out = df[[name_col, team_col]].copy()
    out.columns = ["Person", "Department/Team"]
    out["Person"] = out["Person"].apply(canonicalize_person)
    out["Department/Team"] = out["Department/Team"].apply(canonicalize_department)
    out = out.dropna(subset=["Person"])
    out = out[out["Person"].str.strip() != ""]
    return out.reset_index(drop=True)


def find_zero_loggers(roster_df: pd.DataFrame, jira_pivot: pd.DataFrame) -> pd.DataFrame:
    """
    Return roster members whose canonicalized name does NOT appear in jira_pivot.
    Result columns: ['Department/Team', 'Person']
    """
    if roster_df is None or roster_df.empty:
        return pd.DataFrame(columns=["Department/Team", "Person"])

    logged_names = set(jira_pivot["Person"].dropna().str.strip().str.lower().tolist()) \
        if not jira_pivot.empty else set()

    missing = roster_df[
        ~roster_df["Person"].str.strip().str.lower().isin(logged_names)
    ].copy()

    missing = missing.sort_values(["Department/Team", "Person"])
    return missing.reset_index(drop=True)


def draw_zero_logger_chart(zero_df: pd.DataFrame, title: str = "No JIRA Time Logged This Week") -> bytes:
    """
    Horizontal bar chart showing 0 h bars for each missing person,
    grouped by department – makes the gap visually obvious in the report.
    """
    if zero_df is None or zero_df.empty:
        fig = plt.figure(figsize=(6, 2))
        plt.text(0.5, 0.5, "✓ Everyone on the roster logged JIRA time!", ha="center", va="center", fontsize=11)
        plt.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    df = zero_df.copy()
    df["Total Hours"] = 0.0

    # Build visual rows (dept headers + people)
    dept_order = df["Department/Team"].unique().tolist()
    rows = []
    header_idx = []
    idx = 0
    for dept in dept_order:
        rows.append({"Display": dept, "Value": 0.0, "is_header": True})
        header_idx.append(idx)
        idx += 1
        for person in df[df["Department/Team"] == dept]["Person"].tolist():
            rows.append({"Display": f"  {person}", "Value": 0.0, "is_header": False})
            idx += 1

    vis = pd.DataFrame(rows)
    fig_h = max(3, 0.42 * len(vis))
    fig, ax = plt.subplots(figsize=(10, fig_h))

    y = np.arange(len(vis))
    bar_color = "#d9534f"
    bars = ax.barh(y, vis["Value"].values, color=bar_color, alpha=0.7)

    # Add a tiny stub so 0-bars are visible
    ax.set_xlim(0, 1)

    ax.set_yticks(y)
    ax.set_yticklabels(vis["Display"].tolist(), fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Hours Logged")
    ax.set_title(title, color="#c0392b", fontweight="bold")

    # Bold dept headers + separators
    ticks = ax.get_yticklabels()
    for i in header_idx:
        try:
            ticks[i].set_fontweight("bold")
            ticks[i].set_fontsize(ticks[i].get_fontsize() + 1)
        except Exception:
            pass
        if i != 0:
            ax.axhline(i - 0.5, linewidth=0.6, color="#999")

    # Label each person bar with "0h – not logged"
    for i, row in vis.iterrows():
        if not row["is_header"]:
            ax.text(0.02, i, "0 h — not logged", va="center", ha="left", fontsize=8, color="#c0392b")

    plt.subplots_adjust(left=0.30, right=0.96, top=0.90, bottom=0.06)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def zero_loggers_html_block(zero_df: pd.DataFrame, zero_png_name: str) -> str:
    """Build the Confluence HTML block for the zero-loggers section."""
    if zero_df is None or zero_df.empty:
        return "<h2>✅ All Roster Members Logged JIRA Time</h2><p>No missing entries this week.</p>"

    count = len(zero_df)
    rows_html = ""
    for _, row in zero_df.iterrows():
        rows_html += f"<tr><td>{_esc(row['Department/Team'])}</td><td>{_esc(row['Person'])}</td><td>0</td></tr>"

    img_block = f"""
<ac:image ac:alt="No JIRA Time Logged">
  <ri:attachment ri:filename="{_esc(zero_png_name)}"/>
</ac:image>
"""
    return f"""
<h2>⚠️ No JIRA Time Logged This Week ({count} people)</h2>
{img_block}
<table>
  <colgroup><col/><col/><col/></colgroup>
  <tbody>
    <tr><th>Department/Team</th><th>Person</th><th>JIRA Hours</th></tr>
    {rows_html}
  </tbody>
</table>
""".strip()


# =====================================================================
#  Chart rendering (PNG) – CLUSTERED BY DEPARTMENT + UTIL% ON COMBINED
# =====================================================================

def _dept_utilization_map(pivot_df: pd.DataFrame, capacity_per_person_hours: float) -> dict:
    """
    Util% per dept = dept_total_hours / (capacity_per_person_hours * headcount) * 100
    headcount is # of distinct people shown for that dept in pivot_df.
    """
    if pivot_df is None or pivot_df.empty:
        return {}

    df = pivot_df.copy()
    df["Department/Team"] = df["Department/Team"].fillna("Unassigned/Unknown")
    df["Person"] = df["Person"].fillna("Unassigned/Unknown")
    df["Total Hours"] = pd.to_numeric(df["Total Hours"], errors="coerce").fillna(0.0)

    dept_total = df.groupby("Department/Team", as_index=True)["Total Hours"].sum()
    dept_headcount = df.groupby("Department/Team", as_index=True)["Person"].nunique().replace(0, np.nan)

    util = (dept_total / (capacity_per_person_hours * dept_headcount)) * 100.0
    util = util.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return util.to_dict()


def draw_bar_chart(
    pivot_df: pd.DataFrame,
    title: str,
    show_dept_utilization: bool = False,
    capacity_per_person_hours: float = WEEKLY_CAPACITY_HOURS,
) -> bytes:
    """
    Horizontal bar chart grouped by Department clusters:

    - Department shown as a bold header row on the y-axis with NO bar.
    - Employees of that department listed underneath (indented), each with a bar.
    - Clusters visually separated with a horizontal line.
    - Numeric labels on bars (inside for large bars, just outside for small ones).
    - If show_dept_utilization=True, show utilization % on the dept header row (LEFT aligned).
    """
    if pivot_df is None or pivot_df.empty:
        fig = plt.figure(figsize=(6, 2))
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    df = pivot_df.copy()
    df["Department/Team"] = df["Department/Team"].fillna("Unassigned/Unknown")
    df["Person"] = df["Person"].fillna("Unassigned/Unknown")
    df["Total Hours"] = pd.to_numeric(df["Total Hours"], errors="coerce").fillna(0.0)

    util_map = _dept_utilization_map(df, capacity_per_person_hours) if show_dept_utilization else {}

    # Order departments by total hours (desc)
    dept_tot = df.groupby("Department/Team", as_index=False)["Total Hours"].sum()
    dept_order = dept_tot.sort_values("Total Hours", ascending=False)["Department/Team"].tolist()
    df["__dept_order__"] = pd.Categorical(df["Department/Team"], categories=dept_order, ordered=True)
    df = df.sort_values(["__dept_order__", "Total Hours", "Person"], ascending=[True, False, True]).drop(columns="__dept_order__")

    # Build visual rows: header + members
    rows = []
    header_idx = []
    idx = 0
    for dept, g in df.groupby("Department/Team", sort=False):
        pct = util_map.get(dept, None)
        rows.append({"Display": f"{dept}", "Value": np.nan, "is_header": True, "util_pct": pct})
        header_idx.append(idx)
        idx += 1
        for _, r in g.iterrows():
            rows.append({
                "Display": f"  {r['Person']}",
                "Value": float(r["Total Hours"]) if pd.notna(r["Total Hours"]) else 0.0,
                "is_header": False,
                "util_pct": None,
            })
            idx += 1

    vis = pd.DataFrame(rows)

    fig_h = max(4, 0.42 * len(vis))
    fig, ax = plt.subplots(figsize=(12, fig_h))

    y = np.arange(len(vis))
    values = vis["Value"].fillna(0.0).values
    bars = ax.barh(y, values)

    ax.set_yticks(y)
    ax.set_yticklabels(vis["Display"].tolist(), fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Hours")
    ax.set_title(title)

    # Style dept headers & separators
    ticks = ax.get_yticklabels()
    for i in header_idx:
        try:
            ticks[i].set_fontweight("bold")
            ticks[i].set_fontsize(ticks[i].get_fontsize() + 1)
        except Exception:
            pass
        if i != 0:
            ax.axhline(i - 0.5, linewidth=0.6)

    # Numeric labels on person bars only
    max_val = float(np.nanmax(values)) if len(values) else 0.0
    for i, bar in enumerate(bars):
        if bool(vis.iloc[i]["is_header"]):
            continue
        v = float(vis.iloc[i]["Value"])
        if v <= 0:
            continue
        inside_ok = v >= 0.12 * (max_val if max_val > 0 else 1)
        if inside_ok:
            ax.text(
                bar.get_width() - 0.01 * (max_val if max_val > 0 else 1),
                bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}",
                va="center",
                ha="right",
                color="white",
                fontsize=8,
            )
        else:
            ax.text(
                bar.get_width() + 0.01 * (max_val if max_val > 0 else 1),
                bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}",
                va="center",
                ha="left",
                fontsize=8,
            )

    # Utilization % on header rows (LEFT aligned)
    if show_dept_utilization:
        x_text = (max_val * 0.02) if max_val > 0 else 0.02
        for i in header_idx:
            pct = vis.iloc[i].get("util_pct", None)
            if pct is None:
                continue
            ax.text(
                x_text,
                i,
                f"Util: {pct:.0f}%",
                va="center",
                ha="left",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xlim(left=0, right=(max_val * 1.12 if max_val > 0 else 1))
    plt.subplots_adjust(left=0.30, right=0.96, top=0.92, bottom=0.06)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# =====================================================================
#  Excel export (same as before)
# =====================================================================

def to_excel_with_dashboards(sn_pivot, jira_pivot, combined_pivot) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as w:
        sn_pivot.sort_values(["Total Hours", "Department/Team", "Person"], ascending=[False, True, True]) \
            .to_excel(w, "ServiceNow Pivot", index=False)
        jira_pivot.sort_values(["Total Hours", "Department/Team", "Person"], ascending=[False, True, True]) \
            .to_excel(w, "JIRA Pivot", index=False)
        combined_pivot.sort_values(["Total Hours", "Department/Team", "Person"], ascending=[False, True, True]) \
            .to_excel(w, "Combined Pivot", index=False)
    return bio.getvalue()


# =====================================================================
#  Confluence API helpers (unchanged)
# =====================================================================

def _check_ok(resp: requests.Response, msg: str):
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"{msg}: {resp.status_code} {resp.text[:500]}")


def upload_attachment(base: str, auth: tuple, page_id: str,
                      filename: str, filebytes: bytes,
                      mime: str = "application/octet-stream"):
    base = base.rstrip("/")
    headers = {"X-Atlassian-Token": "no-check"}
    files = {"file": (filename, io.BytesIO(filebytes), mime)}

    create_url = f"{base}/rest/api/content/{page_id}/child/attachment"
    r = requests.post(create_url, auth=auth, headers=headers, files=files, timeout=60)

    def needs_update(resp):
        if resp.status_code in (409,):
            return True
        if resp.status_code == 400:
            try:
                data = resp.json()
                msg = (data.get("message") or "").lower()
                return "same file name" in msg or "already exists" in msg
            except Exception:
                return False
        return False

    if needs_update(r):
        find_url = f"{base}/rest/api/content/{page_id}/child/attachment?filename={requests.utils.quote(filename)}&expand=version"
        fr = requests.get(find_url, auth=auth, timeout=60)
        _check_ok(fr, "Find attachment failed")
        results = fr.json().get("results", [])
        if not results:
            raise RuntimeError("Attachment exists but not found via search.")
        attach_id = results[0]["id"]
        update_url = f"{base}/rest/api/content/{page_id}/child/attachment/{attach_id}/data"
        r = requests.post(update_url, auth=auth, headers=headers, files=files, timeout=60)

    _check_ok(r, f"Upload failed for {filename}")


def get_page_meta(base: str, auth: tuple, page_id: str):
    url = f"{base.rstrip('/')}/rest/api/content/{page_id}?expand=version,title"
    r = requests.get(url, auth=auth, timeout=60)
    _check_ok(r, "GET page failed")
    j = r.json()
    return j["title"], j["version"]["number"]


def update_page_body(base: str, auth: tuple, page_id: str, html: str, new_title: Optional[str] = None):
    title, ver = get_page_meta(base, auth, page_id)
    if new_title:
        title = new_title
    url = f"{base.rstrip('/')}/rest/api/content/{page_id}"
    payload = {
        "id": page_id,
        "type": "page",
        "title": title,
        "version": {"number": ver + 1},
        "body": {"storage": {"value": html, "representation": "storage"}},
    }
    r = requests.put(url, json=payload, auth=auth, timeout=60)
    _check_ok(r, "Page update failed")


def find_child_page_by_title(base: str, auth: tuple, parent_id: str, title: str) -> Optional[str]:
    url = f"{base.rstrip('/')}/rest/api/content/{parent_id}/child/page?limit=500"
    r = requests.get(url, auth=auth, timeout=60)
    _check_ok(r, "List children failed")
    for item in r.json().get("results", []):
        if item.get("title", "") == title:
            return item.get("id")
    return None


def create_child_page(base: str, auth: tuple, parent_id: str, title: str, html: str,
                      space_key: str = CONFLUENCE_SPACE_KEY) -> str:
    url = f"{base.rstrip('/')}/rest/api/content"
    payload = {
        "type": "page",
        "title": title,
        "ancestors": [{"id": parent_id}],
        "space": {"key": space_key},
        "body": {"storage": {"value": html, "representation": "storage"}},
    }
    r = requests.post(url, json=payload, auth=auth, timeout=60)
    _check_ok(r, "Create page failed")
    return r.json()["id"]


# =====================================================================
#  Confluence HTML builders (unchanged)
# =====================================================================

def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&#39;"))


def df_to_html_table(df: pd.DataFrame, header=True) -> str:
    cols = list(df.columns)
    out = ["<table><colgroup>"]
    for _ in cols:
        out.append("<col/>")
    out.append("</colgroup><tbody>")
    if header:
        out.append("<tr>")
        for c in cols:
            out.append(f"<th>{_esc(c)}</th>")
        out.append("</tr>")
    for _, row in df.iterrows():
        out.append("<tr>")
        for c in cols:
            v = row[c]
            out.append(f"<td>{_esc('' if pd.isna(v) else v)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def section_image_block(attachment_filename: str, heading: str) -> str:
    return f"""
<h3>{_esc(heading)}</h3>
<ac:image ac:alt="{_esc(heading)}">
  <ri:attachment ri:filename="{_esc(attachment_filename)}"/>
</ac:image>
""".strip()


def build_page_html(
    jira_png_name: str,
    sn_png_name: str,
    comb_png_name: str,
    jira_pivot: pd.DataFrame,
    sn_pivot: pd.DataFrame,
    combined_pivot: pd.DataFrame,
    zero_df: Optional[pd.DataFrame] = None,
    zero_png_name: Optional[str] = None,
) -> str:
    jira_tbl = df_to_html_table(jira_pivot)
    sn_tbl = df_to_html_table(sn_pivot)
    comb_tbl = df_to_html_table(combined_pivot)

    # NEW (app8): zero-logger section (only if roster was provided)
    zero_section = ""
    if zero_df is not None and zero_png_name is not None:
        zero_section = zero_loggers_html_block(zero_df, zero_png_name)

    return f"""
<h2>Time Utilization Dashboard</h2>

<h3>Dashboards</h3>
{section_image_block(jira_png_name, "Hours per Person (JIRA)")}
{section_image_block(sn_png_name, "Hours per Person (ServiceNow)")}
{section_image_block(comb_png_name, "Hours per Person (JIRA+ServiceNow)")}

<h3>Pivots</h3>
<h4>Hours per Person (JIRA)</h4>
{jira_tbl}
<h4>Hours per Person (ServiceNow)</h4>
{sn_tbl}
<h4>Hours per Person (Combined)</h4>
{comb_tbl}

{zero_section}
""".strip()


# =====================================================================
#  UI: Sidebar inputs (unchanged)
# =====================================================================

st.sidebar.header("Confluence Configuration")
base_url = st.sidebar.text_input(
    "Confluence URL (must end with /wiki)",
    value=_secret("CONFLUENCE_URL", "https://hearsttech.atlassian.net/wiki")
)
parent_page_id = st.sidebar.text_input(
    "Parent Page ID (e.g., 'Time Utilization Report' folder)",
    value=_secret("PARENT_PAGE_ID", "")
)
user = st.sidebar.text_input(
    "Email/User",
    value=_secret("CONFLUENCE_USER", "")
)
token = st.sidebar.text_input(
    "API Token",
    value=_secret("CONFLUENCE_TOKEN", ""),
    type="password"
)
st.sidebar.caption(
    "**Tip — Streamlit secrets:** create `.streamlit/secrets.toml` in your project folder "
    "(or paste into Streamlit Cloud → Settings → Secrets) to pre-fill these fields:\n\n"
    "```toml\n"
    'CONFLUENCE_URL = "https://hearsttech.atlassian.net/wiki"\n'
    'PARENT_PAGE_ID = "YOUR_NEW_PAGE_ID"\n'
    'CONFLUENCE_SPACE_KEY = "HTSIO"\n'
    'CONFLUENCE_USER = "you@hearst.com"\n'
    "```\n\n"
    "You renamed the folder from *Utilization Report 2025* → **Time Utilization Report**. "
    "Make sure to update `PARENT_PAGE_ID` to the new folder's Confluence page ID."
)

# Optional debug toggle
debug_mapping = st.sidebar.checkbox("Debug: show unmapped departments", value=False)

# ── NEW (app8): Roster upload ──────────────────────────────────────
st.sidebar.divider()
st.sidebar.header("👥 Roster (optional)")
st.sidebar.caption(
    "Upload a CSV or XLSX with your team roster to see who logged **zero** JIRA hours this week. "
    "Required columns: **Name** (or Full Name / Person) + **Team** (or Department / Group)."
)
roster_file = st.sidebar.file_uploader(
    "Upload roster file (CSV or XLSX)",
    type=["csv", "xlsx"],
    key="roster_uploader",
)

roster_df: pd.DataFrame = pd.DataFrame(columns=["Person", "Department/Team"])
if roster_file is not None:
    try:
        roster_df = load_roster(roster_file.read(), roster_file.name)
        st.sidebar.success(f"✅ Roster loaded: {len(roster_df)} people across "
                           f"{roster_df['Department/Team'].nunique()} teams.")
    except Exception as e:
        st.sidebar.error(f"Roster error: {e}")


# =====================================================================
#  Main UI
# =====================================================================

st.title("Time Utilization Analyzer → Create Weekly Confluence Page")

uploaded = st.file_uploader(
    "Drop your Excel files here (JIRA + ServiceNow). The app auto-detects which is which.",
    type=["xlsx"],
    accept_multiple_files=True
)

if uploaded:
    jira_pivot = pd.DataFrame(columns=["Department/Team", "Person", "Total Hours"])
    sn_pivot = pd.DataFrame(columns=["Department/Team", "Person", "Total Hours"])
    summary_lines = []

    with st.spinner("Processing files..."):
        for upl in uploaded:
            try:
                content = upl.read()
                xl = pd.ExcelFile(io.BytesIO(content))
                sheet = xl.sheet_names[0]
                df = xl.parse(sheet)

                work_col, person_col, dept_col, _date_col = detect_columns(df)
                platform = detect_platform(df)

                if work_col is None:
                    cands = [c for c in df.columns if any(k in c.lower() for k in ["hour", "time", "duration"])]
                    work_col = cands[0] if cands else None
                    if work_col is None:
                        df["__hours__"] = 0.0
                        work_col = "__hours__"

                df["Hours"] = normalize_hours(df[work_col], platform)
                pivot = build_pivot(df, person_col, dept_col, "Hours")

                if platform == "JIRA":
                    jira_pivot = pd.concat([jira_pivot, pivot], ignore_index=True)
                else:
                    sn_pivot = pd.concat([sn_pivot, pivot], ignore_index=True)

                blanks = int(df[person_col].isna().sum()) if (person_col and person_col in df.columns) else df.shape[0]
                zero = int((df["Hours"] <= 0).sum())
                std = float(df["Hours"].std()) if df["Hours"].std() is not None else 0.0
                outliers = int(df["Hours"][df["Hours"] > (df["Hours"].median() + 3 * (std if std > 0 else 0))].count())

                summary_lines.append(
                    f"{upl.name}: sheet '{sheet}'. work='{work_col}', person='{person_col}', dept='{dept_col}', "
                    f"platform={platform} (units forced {'sec→hr' if platform=='ServiceNow' else 'hr'}); "
                    f"blanks={blanks}, zero_hours={zero}, outliers={outliers}."
                )
            except Exception as e:
                st.error(f"Error processing {upl.name}: {e}")

    # Aggregate duplicates
    if not jira_pivot.empty:
        jira_pivot = jira_pivot.groupby(["Department/Team", "Person"], as_index=False)["Total Hours"].sum()
    if not sn_pivot.empty:
        sn_pivot = sn_pivot.groupby(["Department/Team", "Person"], as_index=False)["Total Hours"].sum()

    # Canonicalize JIRA and SN pivots (names + departments)
    jira_pivot = canonicalize_pivot(jira_pivot)
    sn_pivot = canonicalize_pivot(sn_pivot)

    # Combined: merge then canonicalize + force one-dept-per-person
    combined_raw = pd.concat([jira_pivot, sn_pivot], ignore_index=True)
    combined_pivot = canonicalize_pivot(combined_raw)
    combined_pivot = assign_each_person_to_one_dept(combined_pivot)

    # Debug: reveal unmapped departments BEFORE canonicalization, and what they become AFTER
    if debug_mapping:
        st.subheader("Debug: Department Consolidation")

        # Departments seen in the inputs (post initial pivot, pre-canonicalization)
        # We can’t perfectly reconstruct raw values after canonicalize_pivot,
        # so we show (a) what is in combined_raw and (b) what ends in combined_pivot.
        raw_depts = sorted(pd.Series(combined_raw["Department/Team"].dropna().unique()).tolist())
        final_depts = sorted(pd.Series(combined_pivot["Department/Team"].dropna().unique()).tolist())

        st.write("Departments appearing in combined inputs (after initial canonicalization step):")
        st.code("\n".join(raw_depts) if raw_depts else "None")

        st.write("Departments appearing in final combined chart (after one-dept-per-person):")
        st.code("\n".join(final_depts) if final_depts else "None")

        # Show any departments that still look like they might be duplicates (heuristic)
        suspect = [d for d in final_depts if _norm_key(d).startswith("hts ")]
        if suspect:
            st.warning("These departments still look prefixed (possible missing mapping):\n" + "\n".join(suspect))

    # Display pivots
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Hours per Person (JIRA)")
        st.dataframe(jira_pivot.sort_values(["Department/Team", "Total Hours", "Person"], ascending=[True, False, True]),
                     use_container_width=True)
    with c2:
        st.subheader("Hours per Person (ServiceNow)")
        st.dataframe(sn_pivot.sort_values(["Department/Team", "Total Hours", "Person"], ascending=[True, False, True]),
                     use_container_width=True)
    with c3:
        st.subheader("Hours per Person (Combined)")
        st.dataframe(combined_pivot.sort_values(["Department/Team", "Total Hours", "Person"], ascending=[True, False, True]),
                     use_container_width=True)

    # Render charts (PNG)
    jira_png = draw_bar_chart(jira_pivot, "Hours per Person (JIRA)")
    sn_png = draw_bar_chart(sn_pivot, "Hours per Person (ServiceNow)")
    comb_png = draw_bar_chart(
        combined_pivot,
        "Hours per Person (JIRA+ServiceNow)",
        show_dept_utilization=True,
        capacity_per_person_hours=WEEKLY_CAPACITY_HOURS,
    )

    st.subheader("Dashboards (Preview)")
    colA, colB, colC = st.columns(3)
    with colA:
        st.image(jira_png, caption="Hours per Person (JIRA)")
    with colB:
        st.image(sn_png, caption="Hours per Person (ServiceNow)")
    with colC:
        st.image(comb_png, caption="Hours per Person (JIRA+ServiceNow)")

    # ── NEW (app8): Zero JIRA loggers ─────────────────────────────────
    zero_df = find_zero_loggers(roster_df, jira_pivot)
    zero_png = draw_zero_logger_chart(zero_df)

    st.divider()
    if not roster_df.empty:
        if zero_df.empty:
            st.success("✅ Everyone on the roster logged JIRA time this week — great job!")
        else:
            st.subheader(f"⚠️ No JIRA Time Logged This Week — {len(zero_df)} people")
            st.caption(
                "These roster members have **zero hours** in the JIRA/Tempo report for this period. "
                "This may mean they forgot to log, were on leave, or their name doesn't match the roster exactly."
            )
            st.image(zero_png, caption="No JIRA Time Logged")

            # Display grouped by team
            for dept, grp in zero_df.groupby("Department/Team"):
                with st.expander(f"**{dept}** — {len(grp)} person(s)", expanded=True):
                    st.dataframe(
                        grp[["Person"]].reset_index(drop=True),
                        use_container_width=True,
                        hide_index=True,
                    )
    else:
        st.info(
            "💡 **Tip (new in this version):** Upload a **Roster** file in the sidebar to see who logged zero "
            "JIRA hours this week. Your manager asked for it — now you have it! "
            "The roster just needs two columns: Name + Team."
        )

    # Downloads (optional)
    excel_bytes = to_excel_with_dashboards(sn_pivot, jira_pivot, combined_pivot)
    st.download_button(
        "Download Excel (pivots)",
        data=excel_bytes,
        file_name="Time Utilization Pivots.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # Processing summary
    ai_findings = []
    if jira_pivot.empty and not sn_pivot.empty:
        ai_findings.append("JIRA pivot is empty; likely no JIRA worklogs for this period.")
    if sn_pivot.empty and not jira_pivot.empty:
        ai_findings.append("ServiceNow pivot is empty; likely no SN worklogs for this period.")
    if not ai_findings:
        ai_findings.append("Both JIRA and ServiceNow pivots generated successfully.")

    summary_text = "\n".join(["Assumptions/Choices & Data Quality:"] + summary_lines + ["", "Findings:"] + [f"- {x}" for x in ai_findings])
    st.text_area("Processing Summary", summary_text, height=180)

    st.divider()
    st.subheader("Publish to Confluence")

    btn = st.button("Publish now")
    if btn:
        if not (base_url and parent_page_id and user and token):
            st.error("Confluence URL / Parent Page ID / User / Token are required.")
        else:
            try:
                auth = (user, token)

                child_title = pick_week_title_from_uploads(uploaded)

                existing_child_id = find_child_page_by_title(base_url, auth, parent_page_id, child_title)
                if existing_child_id:
                    child_id = existing_child_id
                else:
                    stub_html = "<h2>Time Utilization Dashboard</h2><p>Publishing…</p>"
                    child_id = create_child_page(
                        base_url, auth, parent_page_id, child_title, stub_html, space_key=CONFLUENCE_SPACE_KEY
                    )

                jira_png_name = "chart_hours_per_person_jira.png"
                sn_png_name = "chart_hours_per_person_servicenow.png"
                comb_png_name = "chart_hours_per_person_combined.png"
                zero_png_name = "chart_zero_jira_loggers.png"

                upload_attachment(base_url, auth, child_id, jira_png_name, jira_png, "image/png")
                upload_attachment(base_url, auth, child_id, sn_png_name, sn_png, "image/png")
                upload_attachment(base_url, auth, child_id, comb_png_name, comb_png, "image/png")
                # NEW (app8): upload zero-logger chart if roster was provided
                if not roster_df.empty:
                    upload_attachment(base_url, auth, child_id, zero_png_name, zero_png, "image/png")
                upload_attachment(
                    base_url, auth, child_id,
                    "Time Utilization Pivots.xlsx", excel_bytes,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

                html = build_page_html(
                    jira_png_name, sn_png_name, comb_png_name,
                    jira_pivot.sort_values(["Department/Team", "Total Hours", "Person"], ascending=[True, False, True]),
                    sn_pivot.sort_values(["Department/Team", "Total Hours", "Person"], ascending=[True, False, True]),
                    combined_pivot.sort_values(["Department/Team", "Total Hours", "Person"], ascending=[True, False, True]),
                    zero_df=zero_df if not roster_df.empty else None,
                    zero_png_name=zero_png_name if not roster_df.empty else None,
                )

                update_page_body(base_url, auth, child_id, html, new_title=child_title)
                st.success(f"Published ✅ → Created/Updated child page: {child_title}")

            except Exception as e:
                st.error(f"Publish failed: {e}")

else:
    st.info("Drop your two Excel files above to get started.")