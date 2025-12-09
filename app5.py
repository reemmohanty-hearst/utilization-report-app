# app5.py  (copy of working app3.py with date-range title + clustered charts)

import io
import os
import re
import json
import time
import tempfile
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import streamlit as st

# ------------- Streamlit page config -------------
st.set_page_config(page_title="Time Utilization Analyzer → Publish to Confluence", layout="wide")

# Hard-default space key for the HTS I&O space (adjust if needed)
CONFLUENCE_SPACE_KEY = "HTSIO"

# =====================================================================
#  Helpers: filename date parsing (for page title)
# =====================================================================

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

def extract_mmdd_from_timetracking_filename(name: str) -> Optional[str]:
    """
    OLD behavior (ServiceNow file), kept as fallback:
      'Time Tracking for I&O teams (List report)- 11/10.xlsx' -> '11/10'
    """
    m = re.search(
        r"Time Tracking for I&O teams\s*\(List report\)\s*-\s*([0-9]{1,2}[/-][0-9]{1,2})",
        name,
        re.I,
    )
    if m:
        mmdd = m.group(1).replace("-", "/")
        parts = mmdd.split("/")
        if len(parts) == 2:
            m_ = parts[0].zfill(2)
            d_ = parts[1].zfill(2)
            return f"{m_}/{d_}"
        return mmdd
    return None

def extract_range_from_jira_filename(name: str) -> Optional[str]:
    """
    NEW behavior: extract MM/DD–MM/DD from a Jira report filename, e.g.:

      'I&O_Report_Only_Projects_03_Nov_25_09_Nov_25.xlsx'
       -> '11/03–11/09'

    Pattern: ..._<DD>_<Mon>_<YY>_<DD>_<Mon>_<YY>...
    """
    # Allow anything before, then 3 parts (D_Mmm_YY) twice.
    # Example: 03_Nov_25_09_Nov_25
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

    # We only care about MM/DD display; year is not shown in title.
    m1 = str(MONTH_MAP[mon1]).zfill(2)
    m2 = str(MONTH_MAP[mon2]).zfill(2)
    d1 = d1.zfill(2)
    d2 = d2.zfill(2)

    return f"{m1}/{d1}–{m2}/{d2}"

def pick_week_title_from_uploads(uploaded_files) -> str:
    """
    PRIORITY:
      1) If ANY uploaded file looks like a Jira report with a date range, use that:
           'Time Utilization Report (MM/DD–MM/DD)'
      2) Else, if we find the older ServiceNow-style Time Tracking file, use its MM/DD:
           'Time Utilization Report (MM/DD)'
      3) Else, fall back to today's date MM/DD.
    """
    date_range = None
    mmdd = None

    for f in uploaded_files or []:
        # First try Jira-style range
        date_range = extract_range_from_jira_filename(f.name)
        if date_range:
            break

    if date_range:
        return f"Time Utilization Report ({date_range})"

    # Fallback to old timetracking pattern
    for f in uploaded_files or []:
        mmdd = extract_mmdd_from_timetracking_filename(f.name)
        if mmdd:
            break

    if not mmdd:
        mmdd = datetime.now().strftime("%m/%d")
    return f"Time Utilization Report ({mmdd})"

# =====================================================================
#  Helpers: column detection & transforms
# =====================================================================

def detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Try to detect 'work time' (or hours), 'person', 'department/team', and a 'created/Date' column (optional).
    """
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(cands: List[str]) -> Optional[str]:
        for key in cands:
            if key in cols:
                return cols[key]
            for c in df.columns:
                if key.replace(" ", "") == c.lower().replace(" ", ""):
                    return c
        return None

    work_col  = pick(["work time", "logged hours", "time spent", "hours", "duration", "time worked", "work_time"])
    person_col= pick(["assigned to", "full name", "assignee", "user", "name", "resource", "owner", "person"])
    dept_col  = pick(["tempo team", "assignment group", "team", "department", "group", "org", "department/team"])
    date_col  = pick(["created", "date", "work date", "updated"])  # optional
    return work_col, person_col, dept_col, date_col

def detect_platform(df: pd.DataFrame) -> str:
    lc = [c.lower() for c in df.columns]
    j = {"epic","story","issue key","issue id","sprint","project key","logged hours","tempo team"}
    s = {"assignment group","assigned to","incident","problem","change","service","work time"}
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
        # SN reports often in seconds → convert to hours
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
    return grp.rename(columns={dept_col:"Department/Team", person_col:"Person", hours_col:"Total Hours"})

# =====================================================================
#  Chart rendering (PNG) – CLUSTERED BY DEPARTMENT
# =====================================================================

def draw_bar_chart(pivot_df: pd.DataFrame, title: str) -> bytes:
    """
    Horizontal bar chart grouped by Department clusters:

    - Department shown as a bold header row on the y-axis with NO bar.
    - Employees of that department listed underneath (indented), each with a bar.
    - Clusters visually separated with a horizontal line.
    - Numeric labels on bars (inside for large bars, just outside for small ones).
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
        # department header row (no bar)
        rows.append({"Display": f"{dept}", "Value": np.nan, "is_header": True})
        header_idx.append(idx)
        idx += 1
        # each employee row
        for _, r in g.iterrows():
            rows.append({
                "Display": f"  {r['Person']}",
                "Value": float(r["Total Hours"]) if pd.notna(r["Total Hours"]) else 0.0,
                "is_header": False,
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

    # Style department header labels and separators
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
    max_val = np.nanmax(values) if len(values) else 0.0
    for i, bar in enumerate(bars):
        if vis.iloc[i]["is_header"]:
            continue  # no numbers on header rows
        v = vis.iloc[i]["Value"]
        if np.isnan(v) or v <= 0:
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

    ax.set_xlim(left=0, right=max_val * 1.12 if max_val > 0 else 1)
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
        sn_pivot.sort_values(["Total Hours","Department/Team","Person"], ascending=[False,True,True]) \
                .to_excel(w, "ServiceNow Pivot", index=False)
        jira_pivot.sort_values(["Total Hours","Department/Team","Person"], ascending=[False,True,True]) \
                 .to_excel(w, "JIRA Pivot", index=False)
        combined_pivot.sort_values(["Total Hours","Department/Team","Person"], ascending=[False,True,True]) \
                       .to_excel(w, "Combined Pivot", index=False)
    return bio.getvalue()

# =====================================================================
#  Confluence API helpers (unchanged)
# =====================================================================

def _check_ok(resp: requests.Response, msg: str):
    if resp.status_code not in (200,201):
        raise RuntimeError(f"{msg}: {resp.status_code} {resp.text[:500]}")

def upload_attachment(base: str, auth: tuple, page_id: str,
                      filename: str, filebytes: bytes,
                      mime: str = "application/octet-stream"):
    """
    Create or update an attachment on a page.
    """
    base = base.rstrip("/")
    headers = {"X-Atlassian-Token": "no-check"}
    files = {"file": (filename, io.BytesIO(filebytes), mime)}

    # try create
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
        # lookup existing
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
    """
    Search direct children of parent_id and return id for the first title match.
    """
    url = f"{base.rstrip('/')}/rest/api/content/{parent_id}/child/page?limit=500"
    r = requests.get(url, auth=auth, timeout=60)
    _check_ok(r, "List children failed")
    for item in r.json().get("results", []):
        if item.get("title","") == title:
            return item.get("id")
    return None

def create_child_page(base: str, auth: tuple, parent_id: str, title: str, html: str,
                      space_key: str = CONFLUENCE_SPACE_KEY) -> str:
    """
    Create a new Confluence child page under the specified parent page.
    MUST include space key for Confluence Cloud.
    """
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
#  Confluence HTML builders (unchanged, just used by page body)
# =====================================================================

def _esc(s: str) -> str:
    return (str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            .replace('"',"&quot;").replace("'", "&#39;"))

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
    # Embed attached PNG so Confluence renders the chart reliably.
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
) -> str:
    """
    Page body: Title stays outside; here we render sections + full pivots.
    Heading requirement: show just 'Time Utilization Dashboard' (no date range).
    """
    jira_tbl = df_to_html_table(jira_pivot.rename(columns={"Department/Team":"Department/Team","Person":"Person","Total Hours":"Total Hours"}))
    sn_tbl = df_to_html_table(sn_pivot.rename(columns={"Department/Team":"Department/Team","Person":"Person","Total Hours":"Total Hours"}))
    comb_tbl = df_to_html_table(combined_pivot.rename(columns={"Department/Team":"Department/Team","Person":"Person","Total Hours":"Total Hours"}))

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
""".strip()

# =====================================================================
#  UI: Sidebar inputs (unchanged – still uses API token)
# =====================================================================

st.sidebar.header("Confluence Configuration")
base_url = st.sidebar.text_input(
    "Confluence URL (must end with /wiki)",
    value=os.getenv("CONFLUENCE_URL", "https://hearsttech.atlassian.net/wiki")
)
parent_page_id = st.sidebar.text_input(
    "Parent Page ID (e.g., 'Utilization Report 2025')",
    value=os.getenv("PARENT_PAGE_ID", "")
)
user = st.sidebar.text_input(
    "Email/User",
    value=os.getenv("CONFLUENCE_USER", "")
)
token = st.sidebar.text_input(
    "API Token",
    value=os.getenv("CONFLUENCE_TOKEN", ""),
    type="password"
)
st.sidebar.caption("Tip: set env vars CONFLUENCE_URL, PARENT_PAGE_ID, CONFLUENCE_USER, CONFLUENCE_TOKEN to prefill.")

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
    jira_pivot = pd.DataFrame(columns=["Department/Team","Person","Total Hours"])
    sn_pivot   = pd.DataFrame(columns=["Department/Team","Person","Total Hours"])
    summary_lines = []

    with st.spinner("Processing files..."):
        for upl in uploaded:
            try:
                content = upl.read()
                xl = pd.ExcelFile(io.BytesIO(content))
                sheet = xl.sheet_names[0]
                df = xl.parse(sheet)

                work_col, person_col, dept_col, date_col = detect_columns(df)
                platform = detect_platform(df)
                if work_col is None:
                    cands = [c for c in df.columns if any(k in c.lower() for k in ["hour","time","duration"])]
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

                blanks  = int(df[person_col].isna().sum()) if (person_col and person_col in df.columns) else df.shape[0]
                zero    = int((df["Hours"] <= 0).sum())
                std = df["Hours"].std() if df["Hours"].std() is not None else 0
                outliers= int(df["Hours"][ df["Hours"] > (df["Hours"].median() + 3*(std if std>0 else 0)) ].count())

                summary_lines.append(
                    f"{upl.name}: sheet '{sheet}'. work='{work_col}', person='{person_col}', dept='{dept_col}', "
                    f"platform={platform} (units forced {'sec→hr' if platform=='ServiceNow' else 'hr'}); "
                    f"blanks={blanks}, zero_hours={zero}, outliers={outliers}."
                )
            except Exception as e:
                st.error(f"Error processing {upl.name}: {e}")

    # Aggregate duplicates (sum across multiple files if needed)
    if not jira_pivot.empty:
        jira_pivot = jira_pivot.groupby(["Department/Team","Person"], as_index=False)["Total Hours"].sum()
    if not sn_pivot.empty:
        sn_pivot   = sn_pivot.groupby(["Department/Team","Person"], as_index=False)["Total Hours"].sum()
    combined_pivot = pd.concat([jira_pivot, sn_pivot], ignore_index=True)
    if not combined_pivot.empty:
        combined_pivot = combined_pivot.groupby(["Department/Team","Person"], as_index=False)["Total Hours"].sum()

    # Display pivots
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Hours per Person (JIRA)")
        st.dataframe(jira_pivot.sort_values(["Total Hours","Department/Team","Person"], ascending=[False,True,True]), use_container_width=True)
    with c2:
        st.subheader("Hours per Person (ServiceNow)")
        st.dataframe(sn_pivot.sort_values(["Total Hours","Department/Team","Person"], ascending=[False,True,True]), use_container_width=True)
    with c3:
        st.subheader("Hours per Person (Combined)")
        st.dataframe(combined_pivot.sort_values(["Total Hours","Department/Team","Person"], ascending=[False,True,True]), use_container_width=True)

    # Render charts (PNG) – now department-clustered
    jira_png = draw_bar_chart(jira_pivot, "Hours per Person (JIRA)")
    sn_png   = draw_bar_chart(sn_pivot, "Hours per Person (ServiceNow)")
    comb_png = draw_bar_chart(combined_pivot, "Hours per Person (JIRA+ServiceNow)")

    st.subheader("Dashboards (Preview)")
    colA, colB, colC = st.columns(3)
    with colA: st.image(jira_png, caption="Hours per Person (JIRA)")
    with colB: st.image(sn_png,   caption="Hours per Person (ServiceNow)")
    with colC: st.image(comb_png, caption="Hours per Person (JIRA+ServiceNow)")

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
    if jira_pivot.empty and not sn_pivot.empty: ai_findings.append("JIRA pivot is empty; likely no JIRA worklogs for this period.")
    if sn_pivot.empty and not jira_pivot.empty: ai_findings.append("ServiceNow pivot is empty; likely no SN worklogs for this period.")
    if not ai_findings: ai_findings.append("Both JIRA and ServiceNow pivots generated successfully.")
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

                # Determine weekly child page title from filenames (Jira date range if available)
                child_title = pick_week_title_from_uploads(uploaded)

                # Find or create the weekly page
                existing_child_id = find_child_page_by_title(base_url, auth, parent_page_id, child_title)
                if existing_child_id:
                    child_id = existing_child_id
                else:
                    # Create a minimal stub first, then we'll update body after attaching images
                    stub_html = "<h2>Time Utilization Dashboard</h2><p>Publishing…</p>"
                    child_id = create_child_page(base_url, auth, parent_page_id, child_title, stub_html, space_key=CONFLUENCE_SPACE_KEY)

                # Attach artifacts (PNGs + optional Excel)
                jira_png_name = "chart_hours_per_person_jira.png"
                sn_png_name   = "chart_hours_per_person_servicenow.png"
                comb_png_name = "chart_hours_per_person_combined.png"
                upload_attachment(base_url, auth, child_id, jira_png_name, jira_png, "image/png")
                upload_attachment(base_url, auth, child_id, sn_png_name,   sn_png,   "image/png")
                upload_attachment(base_url, auth, child_id, comb_png_name, comb_png, "image/png")
                upload_attachment(
                    base_url, auth, child_id,
                    "Time Utilization Pivots.xlsx", excel_bytes,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

                # Build final HTML with embedded images and pivot tables
                html = build_page_html(
                    jira_png_name, sn_png_name, comb_png_name,
                    jira_pivot.sort_values(["Total Hours","Department/Team","Person"], ascending=[False,True,True]),
                    sn_pivot.sort_values(["Total Hours","Department/Team","Person"], ascending=[False,True,True]),
                    combined_pivot.sort_values(["Total Hours","Department/Team","Person"], ascending=[False,True,True]),
                )

                update_page_body(base_url, auth, child_id, html, new_title=child_title)
                st.success(f"Published ✅ → Created/Updated child page: {child_title}")

            except Exception as e:
                st.error(f"Publish failed: {e}")

else:
    st.info("Drop your Excel files above to get started.")