#!/usr/bin/env python3
"""
weekly_utilization_report.py — FULLY AUTOMATED weekly Time Utilization Report
==============================================================================

Headless replacement for the Streamlit app (app-utilization-report-4.py).
Runs every TUESDAY end of day (GitHub Actions cron) with zero human steps:

  1. Computes last week's Monday-Sunday window automatically
  2. Pulls the week's worklogs:
       TEMPO MODE (preferred, TEMPO_API_TOKEN set): exact replica of the saved
         Tempo "logged time" report — same 11 Tempo teams (IDs from the report
         URL), same team attribution, and the zero-logger population comes from
         live Tempo team membership (what the export's "Users" sheet contained)
       JIRA MODE (fallback, no Tempo token): all Jira worklogs in the window,
         scoped + team-attributed via roster.csv
  3. Splits JIRA vs ServiceNow rows (Issue Summary matching ^2\\dQ\\d)
  4. Builds the same three pivots / canonicalization / one-dept-per-person /
     35h utilization / zero-loggers — logic ported verbatim from the app
  5. Renders the same charts + Excel pivot workbook
  6. Creates/updates the weekly child page under Confluence parent 3984720018
     (HTSIO -> Time Utilization Report) and uploads all attachments
  7. Emails the stakeholder list (recipients.txt) in Reem's standard format
     with the new page link — via Office 365 SMTP if configured; otherwise the
     ready-to-send email is saved to out/email.html for one-click sending

Required GitHub Actions secrets:
    ATLASSIAN_EMAIL             reem.mohanty@hearst.com
    ATLASSIAN_API_TOKEN         id.atlassian.com token (Jira + Confluence)
    CONFLUENCE_PARENT_PAGE_ID   3984720018
Optional:
    TEMPO_API_TOKEN             Tempo > Settings > API Integration (enables
                                exact-replica mode; retires roster.csv upkeep)
    SMTP_PASSWORD               enables automatic email sending
    REPORT_WEEK_START           YYYY-MM-DD backfill override
    DRY_RUN=1                   build everything, publish/email nothing

Exit 0 = success; non-zero = failure (GitHub emails on failure).
"""

import io
import os
import re
import sys
import csv
import json
import time
from datetime import datetime, date, timedelta
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import requests

# ------------- Configuration (env-driven) -------------
JIRA_DOMAIN = os.getenv("JIRA_DOMAIN", "hearsttech.atlassian.net")
JIRA_BASE = f"https://{JIRA_DOMAIN}/rest/api/3"
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL", f"https://{JIRA_DOMAIN}/wiki")
CONFLUENCE_SPACE_KEY = os.getenv("CONFLUENCE_SPACE_KEY", "HTSIO")

ATLASSIAN_EMAIL = os.getenv("ATLASSIAN_EMAIL", "")
ATLASSIAN_API_TOKEN = os.getenv("ATLASSIAN_API_TOKEN", "")
PARENT_PAGE_ID = os.getenv("CONFLUENCE_PARENT_PAGE_ID", "3984720018")

DRY_RUN = os.getenv("DRY_RUN", "0") == "1"
OUT_DIR = os.getenv("OUT_DIR", "out")

WEEKLY_CAPACITY_HOURS = 35.0

SCRIPT_VERSION = "2.1"
try:
    sys.stdout.reconfigure(line_buffering=True)  # logs print in real order
except Exception:
    pass


def _secret(key: str, default: str = "") -> str:
    return os.getenv(key, default)

#  EXCLUDED PERSONS — managed-service / out-of-scope individuals
#  These people will be filtered out of ALL pivots, charts, and reports.
#  Names should be in canonical form (Title Case, e.g. "First Last").
# =====================================================================
EXCLUDED_PERSONS = {
    "Hrithik Rajumashankar",
    "Bob Voelker",
}

# =====================================================================
#  HARDCODED MASTER ROSTER - Jim Bazzano's Organization (57 people)
#  Update this list when people join/leave. Format: (Name, Team)
#  TODO: Replace with SharePoint auto-fetch once IT approves Azure app
# =====================================================================

MASTER_ROSTER = [
    ("Randy Lutchna", "Collaboration & Productivity Applications"),
    ("Liang Chan", "Collaboration & Productivity Applications"),
    ("Philip Eye", "Collaboration & Productivity Applications"),
    ("Eric Stevenson", "Collaboration & Productivity Applications"),
    ("Hasan Siddiqui", "Collaboration & Productivity Applications"),
    ("Mario Ricano", "Collaboration & Productivity Applications"),
    ("Einav Sharon", "Identity"),
    ("Josh Goldwasser", "Identity"),
    ("Angela Beavers", "Identity"),
    ("Mehul Chavda", "Identity"),
    ("Lisa Floyd", "Identity"),
    ("Peter Semenchuk", "Identity"),
    ("Jay Neenan", "Identity"),
    ("Ian Jones", "Endpoint Engineering"),
    ("Jake Ogle", "Endpoint Engineering"),
    ("James Taylor", "Endpoint Engineering"),
    ("Nick Hernandez", "Endpoint Engineering"),
    ("Chris Pilarski", "Endpoint Engineering"),
    ("Toby Riding", "Endpoint Engineering"),
    ("Trevor Jones", "Endpoint Engineering"),
    ("Pablo Cabada", "Endpoint Engineering"),
    ("Mark Dowden", "Directory Services"),
    ("Ali Hussain", "Directory Services"),
    ("Marek Charytonowicz", "Directory Services"),
    ("Dennis Wahmann", "Directory Services"),
    ("Jason Crawford", "SRE"),
    ("David McDonald", "SRE"),
    ("Mark Pasechnick", "SRE"),
    ("Josh Welborn", "SRE"),
    ("Sikandar Cheema", "SRE"),
    ("Scott Rychnowski", "SRE"),
    ("Charlie Yi", "Network"),
    ("Burt Werthner", "Network"),
    ("Pat McCaskill", "Network"),
    ("Marlie Joseph", "Network"),
    ("Sean Rutter", "Network"),
    ("Dave Coble", "Network"),
    ("Jay Yaldor", "Network"),
    ("ElHadji Diallo", "Network"),
    ("Rich Timmins", "Network"),
    ("Jason Boulware", "Cloud Engineering"),
    ("David Pickwell", "Cloud Engineering"),
    ("Dev Patel", "Cloud Engineering"),
    ("Jonathan Mims", "Cloud Engineering"),
    ("Jane Lee", "Cloud Engineering"),
    ("Steven Craig", "CCOE"),
    ("Mani Kondamadugula", "CCOE"),
    ("Christopher Raczynsky", "CCOE"),
    ("Jagannathan Jayagopal", "CCOE"),
    ("Carolyn Zakrzewski", "CCOE"),
    ("Marnie Cunnington", "CCOE"),
    ("Lainie Klein", "Project & Portfolio Management"),
    ("Reem Mohanty", "Project & Portfolio Management"),
    ("Marty Monaco", "Project & Portfolio Management"),
    ("Dave Mitchell", "Project & Portfolio Management"),
    ("Alvin Autar", "Cloud Transformation"),
    ("Cynthia Hartley-Cisco", "Cloud Transformation"),
]

def get_hardcoded_roster() -> pd.DataFrame:
    """
    Return the hardcoded master roster as a DataFrame.
    When someone joins/leaves, edit the MASTER_ROSTER list above.
    TODO: Replace with SharePoint auto-fetch once IT approves Azure app.
    """
    df = pd.DataFrame(MASTER_ROSTER, columns=["Person", "Department/Team"])
    # Apply same canonicalization as the time logs so names match
    df["Person"] = df["Person"].apply(canonicalize_person)
    df["Department/Team"] = df["Department/Team"].apply(canonicalize_department)
    return df


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

    # Cloud Engineering (formerly Cloud Automation + CloudOps)
    "cloud engineering": "Cloud Engineering",
    "cloud automation": "Cloud Engineering",
    "cloudops": "Cloud Engineering",
    "cloud ops": "Cloud Engineering",
    "hts cloud automation": "Cloud Engineering",
    "hts cloudops": "Cloud Engineering",
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

    # Remove excluded persons (managed services, out-of-scope)
    df = df[~df["Person"].isin(EXCLUDED_PERSONS)]

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
#  NEW (app8): Zero-logger detection — reads the Tempo "Users" sheet
#  automatically from the JIRA file. No separate roster needed.
# =====================================================================

    out = out[out["Person"].str.strip() != ""]
    return out.reset_index(drop=True)


def find_zero_loggers_from_roster(roster_df: pd.DataFrame, jira_pivot: pd.DataFrame) -> pd.DataFrame:
    """Fallback: compare manual roster against jira_pivot to find zero-loggers."""
    if roster_df is None or roster_df.empty:
        return pd.DataFrame(columns=["Department/Team", "Person"])
    logged_names = set(jira_pivot["Person"].dropna().str.strip().str.lower().tolist()) \
        if not jira_pivot.empty else set()
    missing = roster_df[~roster_df["Person"].str.strip().str.lower().isin(logged_names)].copy()
    return missing.sort_values(["Department/Team", "Person"]).reset_index(drop=True)


def draw_zero_logger_chart(zero_df: pd.DataFrame, title: str = "No JIRA/Tempo Time Logged This Week") -> bytes:
    """Horizontal bar chart showing each person who logged 0h with their status."""
    if zero_df is None or zero_df.empty:
        fig = plt.figure(figsize=(6, 2))
        plt.text(0.5, 0.5, "✓ Everyone logged JIRA/Tempo time this week!", ha="center", va="center", fontsize=11)
        plt.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    people = zero_df["Person"].tolist()
    statuses = zero_df["Timesheet Status"].tolist() if "Timesheet Status" in zero_df.columns \
        else [""] * len(people)
    labels = [f"  {p}  [{s}]" if s and s != "Unknown" else f"  {p}" for p, s in zip(people, statuses)]

    y = np.arange(len(labels))
    fig_h = max(3, 0.45 * len(labels) + 1)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    ax.barh(y, [0.3] * len(y), color="#d9534f", alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Hours Logged")
    ax.set_title(title, color="#c0392b", fontweight="bold", fontsize=11)
    for i in y:
        ax.text(0.35, i, "0 h — not logged", va="center", ha="left", fontsize=8, color="#c0392b")

    plt.subplots_adjust(left=0.38, right=0.96, top=0.90, bottom=0.08)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def zero_loggers_html_block(zero_df: pd.DataFrame, zero_png_name: str, source_label: str = "") -> str:
    """Build the Confluence HTML block for the zero-loggers section."""
    if zero_df is None or zero_df.empty:
        return "<h2>✅ All Team Members Logged JIRA/Tempo Time</h2><p>No missing entries this week.</p>"

    count = len(zero_df)
    has_status = "Timesheet Status" in zero_df.columns
    has_pct = "Progress %" in zero_df.columns
    has_dept = "Department/Team" in zero_df.columns

    # Build header row
    th = ("<th>Department/Team</th>" if has_dept else "") + \
         "<th>Person</th>" + \
         ("<th>Timesheet Status</th>" if has_status else "") + \
         ("<th>Progress %</th>" if has_pct else "") + \
         "<th>Logged Hours</th>"

    rows_html = ""
    for _, row in zero_df.iterrows():
        td = (f"<td>{_esc(row.get('Department/Team', ''))}</td>" if has_dept else "") + \
             f"<td>{_esc(row['Person'])}</td>" + \
             (f"<td>{_esc(row.get('Timesheet Status', ''))}</td>" if has_status else "") + \
             (f"<td>{_esc(str(row.get('Progress %', '')))}</td>" if has_pct else "") + \
             "<td>0</td>"
        rows_html += f"<tr>{td}</tr>"

    src = f"<p><em>Source: {_esc(source_label)}</em></p>" if source_label else ""
    img = f'<ac:image ac:alt="No JIRA Time Logged"><ri:attachment ri:filename="{_esc(zero_png_name)}"/></ac:image>'

    return f"""
<h2>⚠️ No JIRA/Tempo Time Logged This Week ({count} people)</h2>
{src}
{img}
<table><colgroup><col/><col/><col/></colgroup><tbody>
<tr>{th}</tr>
{rows_html}
</tbody></table>
""".strip()


# =====================================================================

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
            .to_excel(w, sheet_name="ServiceNow Pivot", index=False)
        jira_pivot.sort_values(["Total Hours", "Department/Team", "Person"], ascending=[False, True, True]) \
            .to_excel(w, sheet_name="JIRA Pivot", index=False)
        combined_pivot.sort_values(["Total Hours", "Department/Team", "Person"], ascending=[False, True, True]) \
            .to_excel(w, sheet_name="Combined Pivot", index=False)
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
    zero_source: str = "",
) -> str:
    jira_tbl = df_to_html_table(jira_pivot)
    sn_tbl = df_to_html_table(sn_pivot)
    comb_tbl = df_to_html_table(combined_pivot)

    # Zero-logger section — auto from Users sheet, or roster fallback
    zero_section = ""
    if zero_df is not None and zero_png_name is not None:
        zero_section = zero_loggers_html_block(zero_df, zero_png_name, source_label=zero_source)

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

SN_ISSUE_SUMMARY_PATTERN = re.compile(r"^2[0-9]Q\d", re.IGNORECASE)


def is_servicenow_row(issue_summary: str) -> bool:
    """Return True if the Issue Summary indicates a ServiceNow ticket."""
    if issue_summary is None or (isinstance(issue_summary, float) and np.isnan(issue_summary)):
        return False
    return bool(SN_ISSUE_SUMMARY_PATTERN.match(str(issue_summary).strip()))


# =====================================================================
#  NEW — Roster loading from roster.csv (fallback mode only)
# =====================================================================

def load_roster_csv(path: str = "roster.csv") -> pd.DataFrame:
    """Load the master roster from roster.csv (columns: Person, Team).
    Used in Jira-fallback mode; Tempo mode gets its population from
    Tempo team memberships instead. Falls back to the embedded roster."""
    if os.path.exists(path):
        rows = []
        with open(path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                name = (r.get("Person") or r.get("Name") or "").strip()
                team = (r.get("Team") or r.get("Department/Team") or "").strip()
                if name and not name.startswith("#"):
                    rows.append((name, team))
        if rows:
            df = pd.DataFrame(rows, columns=["Person", "Department/Team"])
            print(f"[roster] Loaded {len(df)} people from {path}")
        else:
            df = pd.DataFrame(MASTER_ROSTER, columns=["Person", "Department/Team"])
    else:
        print(f"[roster] {path} not found — using embedded roster")
        df = pd.DataFrame(MASTER_ROSTER, columns=["Person", "Department/Team"])
    df["Person"] = df["Person"].apply(canonicalize_person)
    df["Department/Team"] = df["Department/Team"].apply(canonicalize_department)
    df = df[~df["Person"].isin(EXCLUDED_PERSONS)]
    return df.reset_index(drop=True)


# =====================================================================
#  NEW — Week window calculation (run any day; reports last Mon–Sun)
# =====================================================================

def compute_week_window(today: Optional[date] = None) -> Tuple[date, date]:
    """(monday, sunday) of the most recent COMPLETED week.
       Run Tue 7/21 -> covers Mon 7/13 .. Sun 7/19.
       Override with REPORT_WEEK_START=YYYY-MM-DD for backfills."""
    override = os.getenv("REPORT_WEEK_START", "").strip()
    if override:
        start = datetime.strptime(override, "%Y-%m-%d").date()
        return start, start + timedelta(days=6)
    today = today or date.today()
    this_monday = today - timedelta(days=today.weekday())
    last_monday = this_monday - timedelta(days=7)
    return last_monday, last_monday + timedelta(days=6)


def make_week_title(start: date, end: date) -> str:
    return f"Time Utilization Report ({start.strftime('%m/%d/%y')}–{end.strftime('%m/%d/%y')})"


# =====================================================================
#  Shared HTTP wrapper (retry on 429/5xx) — used for Jira AND Tempo
# =====================================================================

AUTH = None  # (email, api_token) — set in main()
HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}


def safe_request(method: str, url: str, auth=None, headers=None, **kwargs) -> requests.Response:
    for attempt in range(6):
        try:
            resp = requests.request(method, url,
                                    auth=auth if auth is not None else AUTH,
                                    headers=headers or HEADERS,
                                    timeout=90, **kwargs)
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(f"    [retry] network error {e} — waiting {wait}s")
            time.sleep(wait)
            continue
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 5))
            print(f"    [retry] rate limited — waiting {wait}s")
            time.sleep(wait)
            continue
        if resp.status_code >= 500:
            wait = 2 ** attempt
            print(f"    [retry] HTTP {resp.status_code} — waiting {wait}s")
            time.sleep(wait)
            continue
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code} for {url}: {resp.text[:400]}")
        return resp
    raise RuntimeError(f"Too many retries for {url}")


# =====================================================================
#  DATA SOURCE A (preferred) — Tempo API, scoped to the SAME 11 Tempo
#  teams as the saved report Reem downloads today. Exact replica:
#  same population, same team attribution, and zero-loggers come from
#  Tempo team membership (the old export's "Users" sheet, live).
# =====================================================================

TEMPO_BASE = os.getenv("TEMPO_BASE", "https://api.tempo.io/4")
TEMPO_API_TOKEN = os.getenv("TEMPO_API_TOKEN", "")
# From Reem's saved Tempo report URL (logged-time report, 11 teams):
TEMPO_TEAM_IDS = [t.strip() for t in os.getenv(
    "TEMPO_TEAM_IDS", "37,36,16,40,47,10,46,34,13,15,12").split(",") if t.strip()]


def _tempo_headers() -> dict:
    return {"Authorization": f"Bearer {TEMPO_API_TOKEN}", "Accept": "application/json"}


def _tempo_get_paginated(url: str, params: dict) -> List[dict]:
    out: List[dict] = []
    while url:
        resp = safe_request("GET", url, auth=(), headers=_tempo_headers(), params=params)
        data = resp.json()
        out.extend(data.get("results", []))
        url = (data.get("metadata") or {}).get("next")
        params = {}  # next URL already carries the query string
    return out


def resolve_jira_display_names(account_ids: List[str]) -> Dict[str, str]:
    """accountId -> displayName via Jira's bulk user endpoint (200/batch)."""
    names: Dict[str, str] = {}
    ids = [a for a in dict.fromkeys(account_ids) if a]
    for i in range(0, len(ids), 90):
        chunk = ids[i:i + 90]
        resp = safe_request("GET", f"{JIRA_BASE}/user/bulk",
                            params=[("accountId", a) for a in chunk] + [("maxResults", 200)])
        for u in resp.json().get("values", []):
            names[u.get("accountId", "")] = u.get("displayName", "") or ""
    missing = [a for a in ids if a not in names]
    for a in missing:  # rare fallback, one at a time
        try:
            resp = safe_request("GET", f"{JIRA_BASE}/user", params={"accountId": a})
            names[a] = resp.json().get("displayName", "") or a
        except Exception:
            names[a] = a
    return names


def list_accessible_teams() -> Dict[str, str]:
    """All Tempo teams this token can see: {team_id: name}."""
    results = _tempo_get_paginated(f"{TEMPO_BASE}/teams", {"limit": 200})
    return {str(t.get("id")): (t.get("name") or f"Team {t.get('id')}")
            for t in results}


def tempo_team_members(team_id: str) -> List[str]:
    """Active member accountIds for a Tempo team."""
    results = _tempo_get_paginated(f"{TEMPO_BASE}/teams/{team_id}/members", {})
    ids = []
    for r in results:
        acct = ((r.get("member") or {}).get("accountId")) or ""
        if acct:
            ids.append(acct)
    return ids


def pull_week_worklogs_tempo(start: date, end: date) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (worklog_df, population_df):
      worklog_df   : Full name | Issue Summary | Issue Key | Hours | Work date | Tempo Team
      population_df: Person | Department/Team   (live Tempo team members —
                     replaces roster.csv AND the old export's Users sheet)
    """
    accessible = list_accessible_teams()
    listing = ", ".join(f"{accessible[i]} (id {i})"
                        for i in sorted(accessible, key=lambda x: int(x)))
    print(f"[tempo] Token can see {len(accessible)} teams: {listing}")

    missing = [tid for tid in TEMPO_TEAM_IDS if tid not in accessible]
    for tid in missing:
        print(f"::warning::Tempo team id {tid} is NOT visible to this token — "
              f"skipped. (Stale ID in the saved-report URL, or missing Tempo "
              f"team permission.) Check the accessible-teams list above.")
    if len(missing) > 3:
        raise RuntimeError(
            f"{len(missing)} of {len(TEMPO_TEAM_IDS)} configured Tempo teams are "
            f"not accessible ({missing}) — looks systemic (token permissions?). "
            f"Refusing to publish a badly incomplete report.")

    team_names: Dict[str, str] = {}
    team_members: Dict[str, List[str]] = {}
    raw_logs: List[dict] = []
    seen_worklog_ids = set()

    for tid in TEMPO_TEAM_IDS:
        if tid not in accessible:
            continue
        tname = accessible[tid]
        try:
            members = tempo_team_members(tid)
            logs = _tempo_get_paginated(
                f"{TEMPO_BASE}/worklogs/team/{tid}",
                {"from": start.isoformat(), "to": end.isoformat(), "limit": 1000})
        except Exception as e:
            print(f"::warning::Tempo team {tname} (id {tid}) failed to load and "
                  f"was skipped: {e}")
            continue
        team_names[tid] = tname
        team_members[tid] = members
        fresh = 0
        for wl in logs:
            wid = wl.get("tempoWorklogId")
            if wid in seen_worklog_ids:
                continue  # person on two teams — count the worklog once
            seen_worklog_ids.add(wid)
            wl["_team_id"] = tid
            raw_logs.append(wl)
            fresh += 1
        print(f"[tempo] {tname} (id {tid}): {len(members)} members, "
              f"{fresh} worklogs")

    if not team_names:
        raise RuntimeError("No Tempo teams could be loaded — cannot build report.")

    # Resolve names + issue summaries via Jira
    all_accts = [ (wl.get("author") or {}).get("accountId", "") for wl in raw_logs ]
    for tid in team_members:
        all_accts.extend(team_members[tid])
    acct_names = resolve_jira_display_names(all_accts)

    issue_ids = list({str((wl.get("issue") or {}).get("id", "")) for wl in raw_logs} - {""})
    issues = bulkfetch_issues(issue_ids, fields=["summary"]) if issue_ids else []
    issue_meta = {str(i.get("id")): (i.get("key", ""), (i.get("fields") or {}).get("summary", "") or "")
                  for i in issues}

    rows = []
    for wl in raw_logs:
        acct = (wl.get("author") or {}).get("accountId", "")
        iid = str((wl.get("issue") or {}).get("id", ""))
        key, summary = issue_meta.get(iid, ("", ""))
        rows.append({
            "Full name": acct_names.get(acct, acct),
            "Issue Summary": summary,
            "Issue Key": key,
            "Hours": round((wl.get("timeSpentSeconds", 0) or 0) / 3600.0, 2),
            "Work date": (wl.get("startDate") or "")[:10],
            "Tempo Team": team_names[wl["_team_id"]],
        })

    pop_rows = []
    seen_people = set()
    for tid in team_members:
        for acct in team_members[tid]:
            name = acct_names.get(acct, acct)
            if name in seen_people:
                continue  # first team in the ID list wins for zero-logger table
            seen_people.add(name)
            pop_rows.append({"Person": name, "Department/Team": team_names[tid]})

    return pd.DataFrame(rows), pd.DataFrame(pop_rows)


# =====================================================================
#  DATA SOURCE B (fallback) — Jira REST worklogs + roster.csv scoping.
#  Used automatically when TEMPO_API_TOKEN is not set.
# =====================================================================

def search_issue_ids_with_worklogs(start: date, end: date) -> List[str]:
    jql = f'worklogDate >= "{start.isoformat()}" AND worklogDate <= "{end.isoformat()}"'
    print(f"[jira] JQL: {jql}")
    ids: List[str] = []
    token = None
    while True:
        payload = {"jql": jql, "maxResults": 5000, "fields": ["id"]}
        if token:
            payload["nextPageToken"] = token
        resp = safe_request("POST", f"{JIRA_BASE}/search/jql", json=payload)
        data = resp.json()
        ids.extend(str(i["id"]) for i in data.get("issues", []))
        token = data.get("nextPageToken")
        print(f"[jira]   running total: {len(ids)} issues")
        if not token:
            break
    return ids


def bulkfetch_issues(issue_ids: List[str], fields: Optional[List[str]] = None) -> List[dict]:
    fields = fields or ["summary", "worklog"]
    out: List[dict] = []
    for i in range(0, len(issue_ids), 100):
        chunk = issue_ids[i:i + 100]
        resp = safe_request("POST", f"{JIRA_BASE}/issue/bulkfetch",
                            json={"issueIdsOrKeys": chunk, "fields": fields})
        out.extend(resp.json().get("issues", []))
        print(f"[jira]   bulkfetch {min(i+100, len(issue_ids))}/{len(issue_ids)}")
    return out


def fetch_all_worklogs_for_issue(issue_key: str) -> List[dict]:
    logs, start_at = [], 0
    while True:
        resp = safe_request("GET", f"{JIRA_BASE}/issue/{issue_key}/worklog",
                            params={"startAt": start_at, "maxResults": 100})
        data = resp.json()
        logs.extend(data.get("worklogs", []))
        if start_at + data.get("maxResults", 100) >= data.get("total", 0):
            break
        start_at += data.get("maxResults", 100)
    return logs


def pull_week_worklogs_jira(start: date, end: date) -> pd.DataFrame:
    ids = search_issue_ids_with_worklogs(start, end)
    if not ids:
        return pd.DataFrame(columns=["Full name", "Issue Summary", "Issue Key",
                                     "Hours", "Work date"])
    issues = bulkfetch_issues(ids)
    rows, enriched = [], 0
    for iss in issues:
        key = iss.get("key", "")
        fields = iss.get("fields", {}) or {}
        summary = fields.get("summary", "") or ""
        wl_block = fields.get("worklog", {}) or {}
        logs = wl_block.get("worklogs", []) or []
        if wl_block.get("total", 0) > len(logs):
            logs = fetch_all_worklogs_for_issue(key)
            enriched += 1
        for wl in logs:
            started = (wl.get("started") or "")[:10]
            if not started:
                continue
            d = datetime.strptime(started, "%Y-%m-%d").date()
            if not (start <= d <= end):
                continue
            rows.append({
                "Full name": (wl.get("author") or {}).get("displayName", "") or "",
                "Issue Summary": summary,
                "Issue Key": key,
                "Hours": round((wl.get("timeSpentSeconds", 0) or 0) / 3600.0, 2),
                "Work date": started,
            })
    print(f"[jira] {len(rows)} worklog rows ({len(issues)} issues, {enriched} paginated)")
    return pd.DataFrame(rows)


# =====================================================================
#  NEW — Stakeholder email (matches the Copilot-drafted format Reem sends)
# =====================================================================

EMAIL_FROM = os.getenv("EMAIL_FROM", os.getenv("ATLASSIAN_EMAIL", ""))
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.office365.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", EMAIL_FROM)
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")

SIGNATURE_HTML = """<p>Best,</p>
<p><strong>Reem Mohanty</strong><br>Jr. Project Manager<br>HTS I&amp;O</p>
<p>300 West 57th Street<br>New York, NY 10019<br>M: 4159103034<br>
<a href="mailto:reem.mohanty@hearst.com">reem.mohanty@hearst.com</a></p>"""


def load_recipients(path: str = "recipients.txt") -> List[str]:
    if not os.path.exists(path):
        print(f"[email] {path} not found — no recipients")
        return []
    with open(path) as f:
        return [ln.strip() for ln in f
                if ln.strip() and not ln.strip().startswith("#")]


def build_email(title: str, page_url: str, zero_count: int) -> Tuple[str, str, str]:
    """Returns (subject, html_body, text_body) in Reem's standard format."""
    subject = title
    html = f"""<p>Good afternoon everyone,</p>
<p>Please find last week&rsquo;s Time Utilization Report published in Confluence. The report includes:</p>
<ul>
<li>Automatically calculated time utilization metrics</li>
<li>Department-level grouped bar charts</li>
<li>Pivot tables for detailed breakdowns</li>
</ul>
<p>Additionally, the report also includes those who haven&rsquo;t logged any hours for the week.</p>
<p>Here is the link:<br><a href="{page_url}">{page_url}</a></p>
<p>If you have any questions or need additional insights, feel free to let me know.</p>
{SIGNATURE_HTML}"""
    text = (f"Good afternoon everyone,\n\n"
            f"Please find last week's Time Utilization Report published in Confluence. "
            f"The report includes:\n"
            f"- Automatically calculated time utilization metrics\n"
            f"- Department-level grouped bar charts\n"
            f"- Pivot tables for detailed breakdowns\n\n"
            f"Additionally, the report also includes those who haven't logged any hours "
            f"for the week.\n\nHere is the link:\n{page_url}\n\n"
            f"If you have any questions or need additional insights, feel free to let me know.\n\n"
            f"Best,\n\nReem Mohanty\nJr. Project Manager\nHTS I&O\n\n"
            f"300 West 57th Street\nNew York, NY 10019\nM: 4159103034\n"
            f"reem.mohanty@hearst.com")
    return subject, html, text


def send_email(subject: str, html_body: str, text_body: str,
               recipients: List[str]) -> bool:
    """Send via Office 365 SMTP if credentials are configured. Returns True on send."""
    if not (SMTP_PASSWORD and EMAIL_FROM and recipients):
        print("[email] SMTP not configured (or no recipients) — email NOT sent. "
              "The ready-to-send email is in out/email.html / out/email.txt.")
        return False
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=60) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASSWORD)
        s.sendmail(EMAIL_FROM, recipients, msg.as_string())
    print(f"[email] Sent to {len(recipients)} recipients ✅")
    return True


# =====================================================================
#  Orchestration
# =====================================================================

def run(start: date, end: date) -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    title = make_week_title(start, end)
    print(f"=== {title} ===")

    # ---- 1. Pull worklogs (Tempo mode if token present, else Jira mode) ----
    if TEMPO_API_TOKEN:
        print(f"[mode] TEMPO — exact replica of the saved report "
              f"({len(TEMPO_TEAM_IDS)} teams)")
        df, population_df = pull_week_worklogs_tempo(start, end)
        zero_source = (f"Tempo team membership across {len(TEMPO_TEAM_IDS)} teams "
                       f"({len(population_df)} people)")
    else:
        print("[mode] JIRA fallback — roster.csv defines population & teams")
        df = pull_week_worklogs_jira(start, end)
        population_df = load_roster_csv()
        zero_source = f"Master roster ({len(population_df)} people) — roster.csv"

    if df.empty:
        print("[warn] Zero worklogs pulled — refusing to publish an all-zero week. "
              "Failing loudly so a human checks (Jira/Tempo outage? wrong window?).")
        return 2

    # ---- 2. Canonicalize people; establish population & team attribution ----
    population_df["Person"] = population_df["Person"].apply(canonicalize_person)
    population_df["Department/Team"] = population_df["Department/Team"].apply(canonicalize_department)
    population_df = population_df[~population_df["Person"].isin(EXCLUDED_PERSONS)]
    person_team = dict(zip(population_df["Person"], population_df["Department/Team"]))

    df["Person"] = df["Full name"].apply(canonicalize_person)
    if "Tempo Team" not in df.columns:
        df["Tempo Team"] = df["Person"].map(person_team)

    on_pop = df["Person"].isin(person_team)
    skipped = (df.loc[~on_pop].groupby("Person")["Hours"].sum()
                 .sort_values(ascending=False))
    if len(skipped):
        print(f"[scope] Skipped {len(skipped)} people outside the population "
              f"({skipped.sum():.1f}h). Top: "
              + ", ".join(f"{p} ({h:.0f}h)" for p, h in skipped.head(8).items()))
    df = df[on_pop].copy()

    # ---- 3. Split JIRA vs ServiceNow (identical rule to the app) ----
    sn_mask = df["Issue Summary"].apply(is_servicenow_row)
    df_jira, df_sn = df[~sn_mask].copy(), df[sn_mask].copy()
    print(f"[split] {len(df_jira)} JIRA rows + {len(df_sn)} ServiceNow rows")

    # ---- 4. Pivots — same functions, same order as the app ----
    empty = pd.DataFrame(columns=["Department/Team", "Person", "Total Hours"])
    jira_pivot = canonicalize_pivot(build_pivot(df_jira, "Person", "Tempo Team", "Hours")) \
        if len(df_jira) else empty.copy()
    sn_pivot = canonicalize_pivot(build_pivot(df_sn, "Person", "Tempo Team", "Hours")) \
        if len(df_sn) else empty.copy()

    combined_pivot = pd.concat([jira_pivot, sn_pivot], ignore_index=True)
    combined_pivot = combined_pivot.groupby(["Department/Team", "Person"],
                                            as_index=False)["Total Hours"].sum()
    combined_pivot = assign_each_person_to_one_dept(combined_pivot)

    # ---- 5. Zero loggers vs live population ----
    zero_df = find_zero_loggers_from_roster(population_df, combined_pivot)
    print(f"[zero] {len(zero_df)} people with no time logged")

    # ---- 6. Charts + Excel (verbatim app rendering) ----
    jira_png = draw_bar_chart(jira_pivot, "Hours per Person (JIRA)")
    sn_png = draw_bar_chart(sn_pivot, "Hours per Person (ServiceNow)")
    comb_png = draw_bar_chart(combined_pivot, "Hours per Person (JIRA+ServiceNow)",
                              show_dept_utilization=True,
                              capacity_per_person_hours=WEEKLY_CAPACITY_HOURS)
    zero_png = draw_zero_logger_chart(zero_df)
    excel_bytes = to_excel_with_dashboards(sn_pivot, jira_pivot, combined_pivot)

    for name, blob in [("chart_hours_per_person_jira.png", jira_png),
                       ("chart_hours_per_person_servicenow.png", sn_png),
                       ("chart_hours_per_person_combined.png", comb_png),
                       ("chart_zero_jira_loggers.png", zero_png),
                       ("Time Utilization Pivots.xlsx", excel_bytes)]:
        with open(os.path.join(OUT_DIR, name), "wb") as f:
            f.write(blob)
    print(f"[out] Charts + Excel saved to ./{OUT_DIR}/")

    # ---- 7. Publish to Confluence ----
    page_url = ""
    if DRY_RUN:
        print("[dry-run] Skipping Confluence publish + email. Review ./out/")
        page_url = "https://hearsttech.atlassian.net/wiki/... (dry run — no page created)"
    else:
        if not (CONFLUENCE_URL and PARENT_PAGE_ID and ATLASSIAN_EMAIL and ATLASSIAN_API_TOKEN):
            print("[error] Missing ATLASSIAN_EMAIL / ATLASSIAN_API_TOKEN / "
                  "CONFLUENCE_PARENT_PAGE_ID")
            return 3
        auth = (ATLASSIAN_EMAIL, ATLASSIAN_API_TOKEN)
        child_id = find_child_page_by_title(CONFLUENCE_URL, auth, PARENT_PAGE_ID, title)
        if child_id:
            print(f"[confluence] Updating existing page {child_id}")
        else:
            child_id = create_child_page(CONFLUENCE_URL, auth, PARENT_PAGE_ID, title,
                                         "<h2>Time Utilization Dashboard</h2><p>Publishing…</p>",
                                         space_key=CONFLUENCE_SPACE_KEY)
            print(f"[confluence] Created page {child_id}")

        names = ["chart_hours_per_person_jira.png", "chart_hours_per_person_servicenow.png",
                 "chart_hours_per_person_combined.png", "chart_zero_jira_loggers.png"]
        for n, blob in zip(names, [jira_png, sn_png, comb_png, zero_png]):
            upload_attachment(CONFLUENCE_URL, auth, child_id, n, blob, "image/png")
        upload_attachment(CONFLUENCE_URL, auth, child_id, "Time Utilization Pivots.xlsx",
                          excel_bytes,
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        html = build_page_html(
            names[0], names[1], names[2],
            jira_pivot.sort_values(["Department/Team", "Total Hours", "Person"],
                                   ascending=[True, False, True]),
            sn_pivot.sort_values(["Department/Team", "Total Hours", "Person"],
                                 ascending=[True, False, True]),
            combined_pivot.sort_values(["Department/Team", "Total Hours", "Person"],
                                       ascending=[True, False, True]),
            zero_df=zero_df, zero_png_name=names[3], zero_source=zero_source)
        update_page_body(CONFLUENCE_URL, auth, child_id, html, new_title=title)
        page_url = f"{CONFLUENCE_URL}/pages/viewpage.action?pageId={child_id}"
        print(f"[done] Published ✅  {title}")
        print(f"[done] {page_url}")

    # ---- 8. Stakeholder email ----
    subject, html_body, text_body = build_email(title, page_url, len(zero_df))
    with open(os.path.join(OUT_DIR, "email.html"), "w") as f:
        f.write(f"<!-- To: {', '.join(load_recipients())} -->\n"
                f"<!-- Subject: {subject} -->\n{html_body}")
    with open(os.path.join(OUT_DIR, "email.txt"), "w") as f:
        f.write(f"To: {', '.join(load_recipients())}\nSubject: {subject}\n\n{text_body}")
    if not DRY_RUN:
        send_email(subject, html_body, text_body, load_recipients())
    return 0


def main() -> int:
    global AUTH
    if not (ATLASSIAN_EMAIL and ATLASSIAN_API_TOKEN):
        print("[error] Set ATLASSIAN_EMAIL and ATLASSIAN_API_TOKEN env vars.")
        return 3
    AUTH = (ATLASSIAN_EMAIL, ATLASSIAN_API_TOKEN)
    start, end = compute_week_window()
    try:
        return run(start, end)
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
