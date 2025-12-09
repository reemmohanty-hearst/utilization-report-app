import os
import io
import json
import re
import time
import tempfile
from pathlib import Path
from urllib.parse import quote
from datetime import datetime
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import requests

# -----------------------------
# Streamlit page
# -----------------------------
st.set_page_config(page_title="Time Utilization Analyzer", layout="wide")

# =========================
# Helpers: detection & transforms
# =========================
def detect_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(cands: List[str]):
        for key in cands:
            if key in cols:
                return cols[key]
            for c in df.columns:
                if key.replace(" ", "") == c.lower().replace(" ", ""):
                    return c
        return None

    work_col  = pick(["work time","logged hours","time spent","hours","duration","time worked","work_time"])
    person_col= pick(["assigned to","full name","assignee","user","name","resource","owner"])
    dept_col  = pick(["tempo team","assignment group","team","department","group","org"])
    return work_col, person_col, dept_col

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
        s = s / 3600.0
    return s.round(2)

def build_pivot(df: pd.DataFrame, person_col: str, dept_col: str, hours_col="Hours") -> pd.DataFrame:
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

# =========================
# Charts (PNG) – still generated & attached, but NOT relied upon on the page
# =========================
def draw_grouped_chart(pivot_df: pd.DataFrame, title: str) -> bytes:
    if pivot_df is None or pivot_df.empty:
        fig = plt.figure(figsize=(6, 2))
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.axis("off")
        buf = io.BytesIO(); plt.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
        return buf.getvalue()

    df = pivot_df.copy()
    df["Department/Team"] = df["Department/Team"].fillna("Unassigned/Unknown")
    df["Person"] = df["Person"].fillna("Unassigned/Unknown")

    # order departments by total hours
    dept_tot = df.groupby("Department/Team", as_index=False)["Total Hours"].sum()
    dept_order = dept_tot.sort_values("Total Hours", ascending=False)["Department/Team"].tolist()
    df["__dept_order__"] = pd.Categorical(df["Department/Team"], categories=dept_order, ordered=True)
    df = df.sort_values(["__dept_order__", "Total Hours"], ascending=[True, False]).drop(columns="__dept_order__")

    # add header rows (no bars)
    rows, header_idx = [], []
    i = 0
    for dept, g in df.groupby("Department/Team", sort=False):
        rows.append({"Display": f"{dept}", "Total Hours": np.nan, "is_header": True}); header_idx.append(i); i += 1
        for _, r in g.iterrows():
            rows.append({"Display": f"  {r['Person']}",
                         "Total Hours": float(r["Total Hours"]) if pd.notna(r["Total Hours"]) else 0.0,
                         "is_header": False})
            i += 1
    vis = pd.DataFrame(rows)

    fig_h = max(6, 0.42 * len(vis))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    y = np.arange(len(vis))
    bars = ax.barh(y, vis["Total Hours"].values)

    ax.set_yticks(y)
    ax.set_yticklabels(vis["Display"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel("Hours")
    ax.set_title(title)

    # style headers
    ticks = ax.get_yticklabels()
    for idx in header_idx:
        try:
            ticks[idx].set_fontweight("bold")
            ticks[idx].set_fontsize(ticks[idx].get_fontsize() + 1)
        except Exception:
            pass
        if idx != 0:
            ax.axhline(idx - 0.5, linewidth=0.6)

    # numeric labels
    max_val = np.nanmax(vis["Total Hours"].values) if len(vis) else 0.0
    for i, bar in enumerate(bars):
        value = vis.iloc[i]["Total Hours"]
        if vis.iloc[i]["is_header"] or np.isnan(value) or value <= 0:
            continue
        inside_ok = value >= 0.12 * (max_val if max_val > 0 else 1)
        if inside_ok:
            ax.text(bar.get_width() - (0.01 * (max_val if max_val > 0 else 1)),
                    bar.get_y() + bar.get_height()/2, f"{value:.1f}",
                    va="center", ha="right", fontsize=8, color="white")
        else:
            ax.text(bar.get_width() + (0.01 * (max_val if max_val > 0 else 1)),
                    bar.get_y() + bar.get_height()/2, f"{value:.1f}",
                    va="center", ha="left", fontsize=8)

    ax.set_xlim(left=0, right=max_val * 1.12 if max_val > 0 else 1)
    plt.subplots_adjust(left=0.32, right=0.94)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

# =========================
# Excel export with embedded PNGs (still provided)
# =========================
def to_excel_with_dashboards(sn_pivot, jira_pivot, combined_pivot, charts: List[Tuple[str, bytes]]) -> bytes:
    temp_files: List[str] = []
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as w:
        sn_pivot.sort_values(["Total Hours","Department/Team","Person"], ascending=[False,True,True]).to_excel(w, "ServiceNow Pivot", index=False)
        jira_pivot.sort_values(["Total Hours","Department/Team","Person"], ascending=[False,True,True]).to_excel(w, "JIRA Pivot", index=False)
        combined_pivot.sort_values(["Total Hours","Department/Team","Person"], ascending=[False,True,True]).to_excel(w, "Combined Pivot", index=False)

        wb = w.book
        sheet = wb.add_worksheet("Dashboards")
        sheet.write(0, 0, "Dashboards – Horizontal Bar Charts")
        row = 1
        for title, png_bytes in charts:
            sheet.write(row, 0, title)
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tf.write(png_bytes); tf.flush(); tf.close()
            temp_files.append(tf.name)
            sheet.insert_image(row + 1, 0, tf.name, {"x_scale": 0.9, "y_scale": 0.9})
            row += 25
    for pth in temp_files:
        try: os.remove(pth)
        except Exception: pass
    return bio.getvalue()

# =========================
# Confluence API helpers
# =========================
def upload_attachment(base: str, auth: tuple, page_id: str,
                      filename: str, filebytes: bytes,
                      mime: str = "application/octet-stream"):
    base = base.rstrip("/")
    headers = {"X-Atlassian-Token": "no-check"}
    files = {"file": (filename, io.BytesIO(filebytes), mime)}

    # try create
    create_url = f"{base}/rest/api/content/{page_id}/child/attachment"
    r = requests.post(create_url, auth=auth, headers=headers, files=files)

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
        find_url = f"{base}/rest/api/content/{page_id}/child/attachment?filename={quote(filename)}&expand=version"
        fr = requests.get(find_url, auth=auth)
        fr.raise_for_status()
        results = fr.json().get("results", [])
        if not results:
            raise RuntimeError("Attachment exists but not found via search.")
        attach_id = results[0]["id"]
        update_url = f"{base}/rest/api/content/{page_id}/child/attachment/{attach_id}/data"
        r = requests.post(update_url, auth=auth, headers=headers, files=files)

    if r.status_code not in (200, 201):
        raise RuntimeError(f"Upload failed for {filename}: {r.status_code} {r.text[:500]}")

def get_page_meta(base: str, auth: tuple, page_id: str):
    url = f"{base.rstrip('/')}/rest/api/content/{page_id}?expand=version,title"
    r = requests.get(url, auth=auth)
    if r.status_code != 200:
        raise RuntimeError(f"GET page failed: {r.status_code} {r.text[:300]}")
    j = r.json()
    return j["title"], j["version"]["number"]

def update_page_body(base: str, auth: tuple, page_id: str, html: str):
    title, ver = get_page_meta(base, auth, page_id)
    url = f"{base.rstrip('/')}/rest/api/content/{page_id}"
    payload = {
        "id": page_id,
        "type": "page",
        "title": title,
        "version": {"number": ver + 1},
        "body": {"storage": {"value": html, "representation": "storage"}},
    }
    r = requests.put(url, json=payload, auth=auth)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Page update failed: {r.status_code} {r.text[:500]}")

# =========================
# Confluence content builders (tables + Chart macro)
# =========================
def _html_escape(s: str) -> str:
    return (s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
             .replace('"',"&quot;").replace("'", "&#39;"))

def df_to_html_table(df: pd.DataFrame, header=True) -> str:
    # Render a clean HTML table (Confluence storage accepts HTML tables)
    cols = list(df.columns)
    out = ["<table><colgroup>"]
    for _ in cols:
        out.append("<col/>")
    out.append("</colgroup><tbody>")
    if header:
        out.append("<tr>")
        for c in cols:
            out.append(f"<th>{_html_escape(str(c))}</th>")
        out.append("</tr>")
    for _, row in df.iterrows():
        out.append("<tr>")
        for c in cols:
            v = row[c]
            if pd.isna(v):
                v = ""
            out.append(f"<td>{_html_escape(str(v))}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)

def chart_macro_from_table(table_html: str, title: str, width="1000", height="600") -> str:
    # Confluence Chart macro drawing from the supplied table
    return f"""
<ac:structured-macro ac:name="chart">
  <ac:parameter ac:name="title">{_html_escape(title)}</ac:parameter>
  <ac:parameter ac:name="type">bar</ac:parameter>
  <ac:parameter ac:name="orientation">horizontal</ac:parameter>
  <ac:parameter ac:name="legend">false</ac:parameter>
  <ac:parameter ac:name="width">{_html_escape(width)}</ac:parameter>
  <ac:parameter ac:name="height">{_html_escape(height)}</ac:parameter>
  <ac:parameter ac:name="dataDisplay">value</ac:parameter>
  <ac:rich-text-body>
    {table_html}
  </ac:rich-text-body>
</ac:structured-macro>
""".strip()

def build_section(df: pd.DataFrame, heading: str) -> str:
    """
    Build one section:
      - Chart macro (from a compact table: Label + Hours)
      - Full pivot table below it (as native HTML table)
    """
    # Compact chart table: Label = "Dept — Person"
    if df.empty:
        compact = pd.DataFrame({"Label": [], "Hours": []})
    else:
        compact = df.copy()
        compact["Label"] = compact["Department/Team"].fillna("Unknown") + " — " + compact["Person"].fillna("Unknown")
        compact = compact[["Label","Total Hours"]].rename(columns={"Total Hours":"Hours"})

    chart_table_html = df_to_html_table(compact, header=True)
    chart_html = chart_macro_from_table(chart_table_html, title=heading, width="1200", height="800")

    # Full table
    full_table_html = df_to_html_table(
        df.rename(columns={"Department/Team":"Department/Team","Person":"Person","Total Hours":"Total Hours"}),
        header=True
    )

    return f"""
<h3>{_html_escape(heading)}</h3>
{chart_html}
<p></p>
{full_table_html}
""".strip()

# =========================
# Sidebar config (unchanged)
# =========================
st.sidebar.header("Confluence Configuration")
base_url = st.sidebar.text_input("Confluence URL (must end with /wiki)", value=os.getenv("CONFLUENCE_URL", "https://hearsttech.atlassian.net/wiki"))
page_id  = st.sidebar.text_input("Page ID", value=os.getenv("PAGE_ID", ""))
user     = st.sidebar.text_input("Email/User", value=os.getenv("CONFLUENCE_USER", ""))
token    = st.sidebar.text_input("API Token", value=os.getenv("CONFLUENCE_TOKEN", ""), type="password")
st.sidebar.caption("Tip: set env vars CONFLUENCE_URL, PAGE_ID, CONFLUENCE_USER, CONFLUENCE_TOKEN to prefill.")

# =========================
# Main UI
# =========================
st.title("Time Utilization Analyzer – Drag & Drop + Publish to Confluence")

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

                work_col, person_col, dept_col = detect_columns(df)
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
                    jira_pivot = pd.concat([jira_pivot, pivot]).groupby(["Department/Team","Person"], as_index=False)["Total Hours"].sum()
                else:
                    sn_pivot = pd.concat([sn_pivot, pivot]).groupby(["Department/Team","Person"], as_index=False)["Total Hours"].sum()

                blanks  = int(df[person_col].isna().sum()) if (person_col in df.columns) else df.shape[0]
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

    combined_pivot = pd.concat([jira_pivot, sn_pivot], ignore_index=True)
    if not combined_pivot.empty:
        combined_pivot = combined_pivot.groupby(["Department/Team","Person"], as_index=False)["Total Hours"].sum()

    # Show pivots (app view)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Hours per Person (JIRA)")
        st.dataframe(jira_pivot.sort_values(["Total Hours","Department/Team","Person"], ascending=[False,True,True]), use_container_width=True)
    with c2:
        st.subheader("Hours per Person (ServiceNow)")
        st.dataframe(sn_pivot.sort_values(["Total Hours","Department/Team","Person"], ascending=[False,True,True]), use_container_width=True)
    with c3:
        st.subheader("Hours per Person combined (JIRA+ServiceNow)")
        st.dataframe(combined_pivot.sort_values(["Total Hours","Department/Team","Person"], ascending=[False,True,True]), use_container_width=True)

    # Charts (PNG) – generated for attachments only
    jira_png = draw_grouped_chart(jira_pivot, "Hours per Person (JIRA)")
    sn_png   = draw_grouped_chart(sn_pivot, "Hours per Person (ServiceNow)")
    comb_png = draw_grouped_chart(combined_pivot, "Hours per Person combined (JIRA+ServiceNow)")

    st.subheader("Dashboards (App Preview)")
    colA, colB, colC = st.columns(3)
    with colA: st.image(jira_png, caption="Hours per Person (JIRA)")
    with colB: st.image(sn_png,   caption="Hours per Person (ServiceNow)")
    with colC: st.image(comb_png, caption="Hours per Person combined (JIRA+ServiceNow)")

    # Downloads
    st.subheader("Downloads")
    excel_bytes = to_excel_with_dashboards(
        sn_pivot, jira_pivot, combined_pivot,
        [
            ("Hours per Person (JIRA)", jira_png),
            ("Hours per Person (ServiceNow)", sn_png),
            ("Hours per Person combined (JIRA+ServiceNow)", comb_png),
        ],
    )
    st.download_button(
        "Download Excel (tables + dashboards)",
        data=excel_bytes,
        file_name="Time Utilization Data Analysis and Visualization.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.download_button("Download JIRA pivot CSV", data=jira_pivot.to_csv(index=False).encode("utf-8"),
                       file_name="pivot_hours_per_person_jira.csv", mime="text/csv")
    st.download_button("Download ServiceNow pivot CSV", data=sn_pivot.to_csv(index=False).encode("utf-8"),
                       file_name="pivot_hours_per_person_servicenow.csv", mime="text/csv")
    st.download_button("Download Combined pivot CSV", data=combined_pivot.to_csv(index=False).encode("utf-8"),
                       file_name="pivot_hours_per_person_combined.csv", mime="text/csv")

    # Summary
    ai_findings = []
    if jira_pivot.empty and not sn_pivot.empty: ai_findings.append("JIRA pivot is empty; likely no JIRA worklogs for this period.")
    if sn_pivot.empty and not jira_pivot.empty: ai_findings.append("ServiceNow pivot is empty; likely no SN worklogs for this period.")
    if not ai_findings: ai_findings.append("Both JIRA and ServiceNow pivots generated successfully.")
    summary_text = "\n".join(["Assumptions/Choices & Data Quality:"] + summary_lines + ["", "Findings:"] + [f"- {x}" for x in ai_findings])
    st.text_area("Processing Summary", summary_text, height=180)
    st.download_button("Download processing summary", data=summary_text.encode("utf-8"),
                       file_name="time_utilization_processing_summary.txt", mime="text/plain")

    st.divider()
    st.subheader("Publish to Confluence")

    publish_clicked = st.button("Publish now")
    if publish_clicked:
        try:
            if not (base_url and page_id and user and token):
                st.error("Confluence URL / Page ID / User / Token are required.")
            else:
                auth = (user, token)

                # Attach artifacts (keep this behavior)
                upload_attachment(base_url, auth, page_id,
                                  "Time Utilization Data Analysis and Visualization.xlsx", excel_bytes,
                                  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                upload_attachment(base_url, auth, page_id, "chart_hours_per_person_jira.png", jira_png, "image/png")
                upload_attachment(base_url, auth, page_id, "chart_hours_per_person_servicenow.png", sn_png, "image/png")
                upload_attachment(base_url, auth, page_id, "chart_hours_per_person_combined.png", comb_png, "image/png")
                upload_attachment(base_url, auth, page_id, "pivot_hours_per_person_jira.csv", jira_pivot.to_csv(index=False).encode("utf-8"), "text/csv")
                upload_attachment(base_url, auth, page_id, "pivot_hours_per_person_servicenow.csv", sn_pivot.to_csv(index=False).encode("utf-8"), "text/csv")
                upload_attachment(base_url, auth, page_id, "pivot_hours_per_person_combined.csv", combined_pivot.to_csv(index=False).encode("utf-8"), "text/csv")
                upload_attachment(base_url, auth, page_id, "time_utilization_processing_summary.txt", summary_text.encode("utf-8"), "text/plain")

                # ---- Build page body using Confluence-native tables + Chart macros ----
                jira_sec = build_section(
                    jira_pivot.sort_values(["Department/Team","Total Hours","Person"], ascending=[True,False,True]),
                    "Hours per Person (JIRA)"
                )
                sn_sec = build_section(
                    sn_pivot.sort_values(["Department/Team","Total Hours","Person"], ascending=[True,False,True]),
                    "Hours per Person (ServiceNow)"
                )
                comb_sec = build_section(
                    combined_pivot.sort_values(["Department/Team","Total Hours","Person"], ascending=[True,False,True]),
                    "Hours per Person combined (JIRA+ServiceNow)"
                )

                html = f"""
<h2>Time Utilization Dashboard (Auto-Published)</h2>
<p>Last publish: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<h3>📊 Dashboards</h3>
{jira_sec}
<p></p>
{sn_sec}
<p></p>
{comb_sec}

<h3>📁 Download Full Workbook</h3>
<p>
  <ac:link>
    <ri:attachment ri:filename="Time Utilization Data Analysis and Visualization.xlsx"/>
    <ac:plain-text-link-body><![CDATA[Download Excel Report]]></ac:plain-text-link-body>
  </ac:link>
</p>

<h4>CSV Exports</h4>
<ul>
  <li><ac:link><ri:attachment ri:filename="pivot_hours_per_person_jira.csv"/><ac:plain-text-link-body><![CDATA[pivot_hours_per_person_jira.csv]]></ac:plain-text-link-body></ac:link></li>
  <li><ac:link><ri:attachment ri:filename="pivot_hours_per_person_servicenow.csv"/><ac:plain-text-link-body><![CDATA[pivot_hours_per_person_servicenow.csv]]></ac:plain-text-link-body></ac:link></li>
  <li><ac:link><ri:attachment ri:filename="pivot_hours_per_person_combined.csv"/><ac:plain-text-link-body><![CDATA[pivot_hours_per_person_combined.csv]]></ac:plain-text-link-body></ac:link></li>
</ul>

<ac:panel ac:title="Processing Summary &amp; Data Quality">
  <ac:rich-text-body>
    <pre>{_html_escape(summary_text)}</pre>
  </ac:rich-text-body>
</ac:panel>
"""
                update_page_body(base_url, auth, page_id, html)
                st.success("Published to Confluence ✅ (using native tables + Chart macros)")
        except Exception as e:
            st.error(f"Publish failed: {e}")

else:
    st.info("Drop your two Excel files above to get started.")