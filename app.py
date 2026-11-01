# this is the streamlit dashboard. it doesnt rebuild the model.
# keep app.py, mcf_support_5_health.csv, and requirements.txt in the same folder.

from pathlib import Path
import io
import json
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# if i run this as a .py file, __file__ works. if i run the cell in jupyter it doesnt, so i hardcode the milestone 2 folder.
try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path("/Users/aatifmohammed/Desktop/GA Tech Masters/PRACTICUM/Milestone 2")

BUNDLE_NAME = "mcf_support_5_health.csv"


def _first_json(series):
    if series is None or series.empty:
        return None
    for val in series:
        if isinstance(val, str) and val.strip() and val.strip() not in ("[]", "{}", "nan"):
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                continue
        if isinstance(val, (dict, list)):
            return val
    return None


def unpack_bundle(scores_raw):
    scores = scores_raw.copy()
    timeline_rows = []
    if "timeline_json" in scores.columns:
        for _, row in scores.iterrows():
            raw = row.get("timeline_json")
            if not isinstance(raw, str) or not raw.strip() or raw.strip() == "[]":
                continue
            try:
                items = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                rec = dict(item)
                rec["ent_num"] = row.get("ent_num")
                timeline_rows.append(rec)
        scores = scores.drop(columns=["timeline_json"])
    timeline = pd.DataFrame(timeline_rows)

    themes = _first_json(scores["nlp_themes_json"]) if "nlp_themes_json" in scores.columns else None
    nlp_themes = pd.DataFrame(themes) if isinstance(themes, list) else pd.DataFrame()
    if "nlp_themes_json" in scores.columns:
        scores = scores.drop(columns=["nlp_themes_json"])

    sent = _first_json(scores["sentiment_json"]) if "sentiment_json" in scores.columns else None
    nlp_sent = pd.DataFrame(sent) if isinstance(sent, list) else pd.DataFrame()
    if "sentiment_json" in scores.columns:
        scores = scores.drop(columns=["sentiment_json"])

    model = _first_json(scores["model_json"]) if "model_json" in scores.columns else None
    if not isinstance(model, dict):
        model = {}
    if "model_json" in scores.columns:
        scores = scores.drop(columns=["model_json"])

    linkage_rows = []
    if "linkage_json" in scores.columns:
        for _, row in scores.iterrows():
            raw = row.get("linkage_json")
            if not isinstance(raw, str) or not raw.strip() or raw.strip() == "[]":
                continue
            try:
                items = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                rec = dict(item)
                rec["ent_num"] = row.get("ent_num")
                linkage_rows.append(rec)
        scores = scores.drop(columns=["linkage_json"])
    linkage = pd.DataFrame(linkage_rows)

    if "n_child_sites" not in scores.columns:
        if not linkage.empty and "role" in linkage.columns:
            child_n = (
                linkage.assign(_child=linkage["role"].astype(str).str.strip().eq("Child site"))
                .groupby("ent_num")["_child"].sum()
            )
            scores["n_child_sites"] = scores["ent_num"].map(child_n).fillna(0).astype(int)
        else:
            scores["n_child_sites"] = 0
    else:
        scores["n_child_sites"] = pd.to_numeric(scores["n_child_sites"], errors="coerce").fillna(0).astype(int)

    people_rows = []
    if "people_json" in scores.columns:
        for _, row in scores.iterrows():
            raw = row.get("people_json")
            if not isinstance(raw, str) or not raw.strip() or raw.strip() == "[]":
                continue
            try:
                items = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                rec = dict(item)
                rec["ent_num"] = row.get("ent_num")
                people_rows.append(rec)
        scores = scores.drop(columns=["people_json"])
    people = pd.DataFrame(people_rows)

    disq = pd.DataFrame(model.get("disq") or [])
    iter_df = pd.DataFrame(model.get("iter") or [])
    summary = pd.DataFrame(model.get("summary") or [])
    meta = pd.DataFrame(model.get("meta") or [])
    return scores, timeline, nlp_themes, nlp_sent, disq, iter_df, summary, meta, linkage, people


@st.cache_data
def load_bundle(file_bytes):
    scores_raw = pd.read_csv(io.BytesIO(file_bytes))
    return unpack_bundle(scores_raw)

st.set_page_config(
    page_title="MCF Support & Services Health Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

AXIS = dict(labelFontSize=15, titleFontSize=16, labelLimit=480, titleLimit=520)
LEGEND = dict(labelFontSize=14, titleFontSize=15)

# this just makes the text a bit bigger so its easier to read on a laptop
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    h1 { font-size: 2.45rem !important; font-weight: 700 !important; padding-top: 0.35rem !important; margin-top: 0.2rem !important; }
    h2 { font-size: 1.7rem !important; font-weight: 650 !important; margin-top: 0.45rem !important; margin-bottom: 0.35rem !important; }
    h3 { font-size: 1.32rem !important; font-weight: 650 !important; margin-bottom: 0.2rem !important; }
    .block-container { padding-top: 4.2rem !important; padding-bottom: 1.2rem !important; }
    div[data-testid="stVerticalBlock"] { gap: 0.45rem !important; }
    [data-testid="stMarkdownContainer"] p { font-size: 1.12rem !important; line-height: 1.45; }
    [data-testid="stCaption"] { font-size: 1.05rem !important; }
    .stMetric label { font-size: 1.08rem !important; font-weight: 600 !important; }
    .stMetric [data-testid="stMetricValue"] { font-size: 2.05rem !important; font-weight: 700 !important; }
    .stAlert p { font-size: 1.12rem !important; }
    [data-testid="stSidebar"] { font-size: 1.08rem !important; }
    [data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }
    [data-testid="stFileUploaderDropzone"] {
        min-height: 0 !important;
        padding: 0.2rem 0 !important;
        background: transparent !important;
        border: none !important;
    }
    [data-testid="stFileUploader"] section { padding: 0 !important; }
    .dollar-hero { text-align: right; padding-top: 0.35rem; }
    .dollar-hero .dollar-label { font-size: 1.45rem !important; font-weight: 700 !important; margin: 0 !important; line-height: 1.2 !important; }
    .dollar-hero .dollar-num { font-size: 4.6rem !important; font-weight: 800 !important; margin: 0 !important; line-height: 1.05 !important; }
    .dollar-hero .dollar-cap { font-size: 1.22rem !important; font-weight: 600 !important; margin: 0.25rem 0 0 0 !important; opacity: 0.9; line-height: 1.3 !important; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("MCF Support & Services Health Dashboard")
st.sidebar.caption("Team: Support 5")
st.sidebar.caption("Aatif Ahmed Mohammed")
st.sidebar.markdown("---")

# csv has to be uploaded before anything else loads
uploaded = st.sidebar.file_uploader("CSV", type=["csv"])
if uploaded is None:
    st.title("Account Health and Churn Risk")
    st.write("Upload mcf_support_5_health.csv in the sidebar to load the dashboard.")
    st.stop()

bundle_bytes = uploaded.getvalue()

scores, timeline, nlp_themes, nlp_sent, disq, iter_df, summary, meta, linkage, people = load_bundle(bundle_bytes)

if scores.empty:
    st.error("That csv did not have any account rows.")
    st.stop()

# rename the platform labels so they match what i used in the notebook
if "Platform" in scores.columns:
    scores["Platform"] = scores["Platform"].replace({
        "CONNECT": "Native Michelin CF",
        "Marcus": "Legacy Marcus/ NexTraq",
    })

# ops score is the average of queue / field / voc. it is half of the health score now.
SUB_COLS = [c for c in ["Support_Queue_Subscore", "Field_Deployment_Subscore", "VoC_Sentiment_Subscore"] if c in scores.columns]
if SUB_COLS:
    scores["Ops_Score"] = scores[SUB_COLS].mean(axis=1)

if "Top3_Risk_Drivers" in scores.columns:
    scores["Top3_Risk_Drivers"] = scores["Top3_Risk_Drivers"].replace("Low Operational Friction Observed", "None")

# sidebar is just page picker + the dollar assumption
page = st.sidebar.radio(
    "Pages",
    [
        "Strategic Overview",
        "Operator Overview",
        "Statistical Model Information"
    ]
)

TIER_ORDER = ["Green", "Yellow", "Red"]
TIER_COLORS = ["#28a745", "#ffc107", "#dc3545"]
TIER_LABEL = {"Green": "Green", "Yellow": "Yellow", "Red": "Red"}

# default the operator page to the lowest health score account (the riskiest one we actually scored)
scored_mask = scores["Health_Score"].notna() if "Health_Score" in scores.columns else pd.Series(False, index=scores.index)
if scored_mask.any():
    riskiest_idx = scores.loc[scored_mask, "Health_Score"].idxmin()
    default_account = scores.loc[riskiest_idx, "ent_name"] if "ent_name" in scores.columns else "(portfolio)"
else:
    default_account = scores["ent_name"].iloc[0] if "ent_name" in scores.columns and not scores.empty else "(portfolio)"

# page 1 is the leadership / overview page
if page == "Strategic Overview":
    # this is not real arr from the files. i just multiply accounts by a number i pick for the demo.
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Planning assumption")
    assumed_arr = st.sidebar.number_input(
        "Assumed annual value per account ($)",
        min_value=5000,
        max_value=100000,
        value=25000,
        step=5000,
    )
    st.sidebar.caption(
        "Not from this extract. Released ARR is transformed and cannot be turned back into dollars. "
        "Change this number for the demo."
    )

    n_eval = int(len(scores))
    n_red = int((scores["Risk_Tier"].astype(str).str.strip() == "Red").sum()) if "Risk_Tier" in scores.columns else 0
    n_yellow = int((scores["Risk_Tier"].astype(str).str.strip() == "Yellow").sum()) if "Risk_Tier" in scores.columns else 0
    n_green = int((scores["Risk_Tier"].astype(str).str.strip() == "Green").sum()) if "Risk_Tier" in scores.columns else 0
    dollars_review = n_red * float(assumed_arr)
    dollars_str = f"${dollars_review/1_000_000:.1f}M" if dollars_review >= 1_000_000 else f"${dollars_review:,.0f}"

    head_l, head_r = st.columns([1.7, 1.3])
    with head_l:
        st.title("Account Health and Churn Risk")
        st.markdown(
            "- Jul-Dec 2024 Training data, Jan-Jun 2025 Churn Modeling (21 Churn / 313 accounts).\n"
            "- Health Score = half logit (Avg Days To Close) + half Ops Score. Note: if no close time I put a 25 as default on the logit\n"
            "- Risk Colors are relative on that combined score (Red lowest 25%, Yellow next 25%, Green is the remaining)."
        )
    with head_r:
        st.markdown(
            "<div class='dollar-hero'>"
            "<p class='dollar-label'>Potential $ at risk</p>"
            f"<p class='dollar-num'>{dollars_str}</p>"
            "<p class='dollar-cap'>Assumes Red Accounts Churn</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Accounts Evaluated", f"{n_eval:,}")
    kpi2.metric("Red", n_red)
    kpi3.metric("Yellow", n_yellow)
    kpi4.metric("Green", n_green)

    st.markdown("---")

    # three charts: health, colors, ops lines (same layout as before)
    col_a, col_b, col_c = st.columns([1.5, 1.05, 1.9], gap="small")

    with col_a:
        # histogram of the combined health score (all 313 accounts)
        st.subheader("How Overall \"Health Score\" is Spread")
        health_vals = scores["Health_Score"].dropna() if "Health_Score" in scores.columns else pd.Series()
        if not health_vals.empty:
            min_score = max(0, int(np.floor(health_vals.min())))
            max_score = min(100, int(np.ceil(health_vals.max())) + 1)
            counts, bin_edges = np.histogram(health_vals, bins=8, range=(min_score, max_score))
            hist_df = pd.DataFrame({
                "Health Score Range": [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(counts))],
                "Account Count": counts
            })
            hist_df = hist_df[hist_df["Account Count"] > 0]

            chart_health = alt.Chart(hist_df).mark_bar(color="#0275d8", size=36, cornerRadiusEnd=2).encode(
                x=alt.X("Health Score Range:N", title="Health Score", axis=alt.Axis(labelAngle=0, **AXIS)),
                y=alt.Y("Account Count:Q", title="Accounts", axis=alt.Axis(**AXIS)),
                tooltip=["Health Score Range", "Account Count"]
            ).properties(height=220, padding={"left": 4, "right": 4, "top": 6, "bottom": 4})
            st.altair_chart(chart_health, width="stretch")

    with col_b:
        st.subheader("Risk Groups")
        tier_series = scores["Risk_Tier"].astype(str).str.strip() if "Risk_Tier" in scores.columns else pd.Series()
        tier_counts = tier_series.value_counts().reindex(TIER_ORDER).fillna(0).reset_index()
        tier_counts.columns = ["Risk Tier", "Account Count"]
        tier_counts["Label"] = tier_counts["Risk Tier"]
        bars = alt.Chart(tier_counts).mark_bar(size=32, cornerRadiusEnd=3).encode(
            x=alt.X(
                "Label:N",
                sort=TIER_ORDER,
                title="Risk group",
                axis=alt.Axis(labelAngle=0, **AXIS)
            ),
            y=alt.Y("Account Count:Q", title="Accounts", axis=alt.Axis(**AXIS)),
            color=alt.Color(
                "Label:N",
                scale=alt.Scale(domain=TIER_ORDER, range=TIER_COLORS),
                legend=None
            ),
            tooltip=["Label", "Account Count"]
        )
        labels = alt.Chart(tier_counts).mark_text(dy=-10, fontSize=14, fontWeight="bold").encode(
            x=alt.X("Label:N", sort=TIER_ORDER),
            y=alt.Y("Account Count:Q"),
            text=alt.Text("Account Count:Q", format="d")
        )
        st.altair_chart(
            (bars + labels).properties(height=220, padding={"left": 4, "right": 4, "top": 18, "bottom": 4}),
            width="stretch"
        )

    with col_c:
        st.subheader("Ops Metrics")
        st.caption("Left = more problems. Right = healthier.")
        sub_cols = [c for c in ["Support_Queue_Subscore", "Field_Deployment_Subscore", "VoC_Sentiment_Subscore"] if c in scores.columns]
        if sub_cols:
            sub_dist_list = []
            for sc in sub_cols:
                b_counts, _ = np.histogram(scores[sc].dropna(), bins=10, range=(0, 100))
                for i in range(10):
                    sub_dist_list.append({
                        "score_mid": i * 10 + 5,
                        "Range": f"{i*10}-{(i+1)*10}",
                        "Domain": sc.replace("_Subscore", "").replace("_", " "),
                        "Accounts": int(b_counts[i])
                    })
            sub_df = pd.DataFrame(sub_dist_list)

            lines = alt.Chart(sub_df).mark_line(point=True, strokeWidth=2.6).encode(
                x=alt.X(
                    "score_mid:Q",
                    scale=alt.Scale(domain=[0, 100]),
                    title="Score (higher is better)",
                    axis=alt.Axis(labelAngle=0, tickCount=6, **AXIS)
                ),
                y=alt.Y("Accounts:Q", title="Accounts", axis=alt.Axis(**AXIS)),
                color=alt.Color(
                    "Domain:N",
                    scale=alt.Scale(
                        domain=["Support Queue", "Field Deployment", "VoC Sentiment"],
                        range=["#4dabf7", "#cc5de8", "#22b8cf"]
                    ),
                    legend=alt.Legend(title=None, orient="bottom", direction="horizontal", columns=3, **LEGEND)
                ),
                tooltip=["Domain", "Range", "Accounts"]
            )
            st.altair_chart(
                lines.properties(
                    height=220, padding={"left": 4, "right": 4, "top": 6, "bottom": 4}
                ),
                width="stretch"
            )

    # extra ops histogram, under the original three charts
    st.subheader("How Overall \"Ops Score\" is Spread")
    ops_vals = scores["Ops_Score"].dropna() if "Ops_Score" in scores.columns else pd.Series()
    if not ops_vals.empty:
        ops_min = max(0, int(np.floor(ops_vals.min())))
        ops_max = min(100, int(np.ceil(ops_vals.max())) + 1)
        ops_counts, ops_edges = np.histogram(ops_vals, bins=8, range=(ops_min, ops_max))
        ops_hist_df = pd.DataFrame({
            "Ops Score Range": [f"{int(ops_edges[i])}-{int(ops_edges[i+1])}" for i in range(len(ops_counts))],
            "Account Count": ops_counts
        })
        ops_hist_df = ops_hist_df[ops_hist_df["Account Count"] > 0]

        chart_ops = alt.Chart(ops_hist_df).mark_bar(color="#22b8cf", size=36, cornerRadiusEnd=2).encode(
            x=alt.X("Ops Score Range:N", title="Ops Score", axis=alt.Axis(labelAngle=0, **AXIS)),
            y=alt.Y("Account Count:Q", title="Accounts", axis=alt.Axis(**AXIS)),
            tooltip=["Ops Score Range", "Account Count"]
        ).properties(height=220, padding={"left": 4, "right": 4, "top": 6, "bottom": 4})
        st.altair_chart(chart_ops, width="stretch")

    st.markdown("---")
    # this is the only variable that stayed in the logit
    st.header("Distribution of Avg Days To Close")
    st.caption("Only variable kept in the logit. Churner's had a close time that averaged about 20 days vs 31 for accounts that stayed.")

    vals = scores["Avg_Days_To_Close"].dropna() if "Avg_Days_To_Close" in scores.columns else pd.Series()
    if not vals.empty:
        v_counts, v_edges = np.histogram(vals, bins=8)
        v_df = pd.DataFrame({
            "Days": [f"{int(round(v_edges[i]))}-{int(round(v_edges[i+1]))}" for i in range(len(v_counts))],
            "Accounts": v_counts
        })

        var_chart = alt.Chart(v_df).mark_bar(color="#5bc0de", size=42, cornerRadiusEnd=3).encode(
            x=alt.X(
                "Days:N",
                sort=None,
                title="Avg days to close (the variable used in the model)",
                axis=alt.Axis(labelAngle=0, labelOverlap=False, **AXIS)
            ),
            y=alt.Y("Accounts:Q", title="Accounts", axis=alt.Axis(**AXIS)),
            tooltip=["Days", "Accounts"]
        ).properties(height=300)

        st.altair_chart(var_chart, width="stretch")

    st.markdown("---")
    # lowest health scores first. terminated accounts are left out.
    st.header("2. Top 50 Highest-Risk Active Accounts")
    st.caption("Terminated accounts are left out.")

    top50_cols = [c for c in [
        "ent_name", "Health_Score", "Avg_Days_To_Close", "Risk_Tier",
        "Support_Queue_Subscore", "Field_Deployment_Subscore", "VoC_Sentiment_Subscore", "Ops_Score",
        "Top3_Risk_Drivers", "Account Status", "Platform"
    ] if c in scores.columns]

    top50_src = scores[scores["Health_Score"].notna()].copy()
    if "Account Status" in top50_src.columns:
        top50_src = top50_src[~top50_src["Account Status"].astype(str).str.strip().str.lower().eq("terminated")]
    top50_df = top50_src.sort_values("Health_Score", ascending=True).head(50)[top50_cols]
    if "Risk_Tier" in top50_df.columns:
        top50_df = top50_df.copy()
        top50_df["Risk_Tier"] = top50_df["Risk_Tier"].astype(str).str.strip().map(lambda x: TIER_LABEL.get(x, x))
    if "Top3_Risk_Drivers" in top50_df.columns:
        top50_df = top50_df.rename(columns={"Top3_Risk_Drivers": "NLP Operational Notes"})

    st.dataframe(
        top50_df.style.background_gradient(subset=["Health_Score"], cmap="YlOrRd_r"),
        hide_index=True,
        width="stretch"
    )

    st.markdown("---")

    # sentiment histogram + the theme table from step 7
    st.header("3. Customer Comments and Sentiment")
    st.caption(
        "Analyzed Jul-Dec 2024 comments for every account that has a comment in that window. "
        "I did not include 2025-2026 comments so this stays in the same timeframe as the model."
    )

    col_sent, col_nlp = st.columns(2)

    with col_sent:
        # -1 is negative, +1 is positive. i use vader cutoffs of -0.05 and +0.05
        st.subheader("How Sentiment Scores Are Spread for Accounts")
        sent_src = nlp_sent if not nlp_sent.empty and "Mean_Sentiment_Score" in nlp_sent.columns else scores
        if "Mean_Sentiment_Score" in sent_src.columns:
            sent_data = sent_src["Mean_Sentiment_Score"].dropna()
            counts_s, bin_edges_s = np.histogram(sent_data, bins=20, range=(-1, 1))

            sent_hist_df = pd.DataFrame({
                "bin_start": np.round(bin_edges_s[:-1], 2),
                "bin_end": np.round(bin_edges_s[1:], 2),
                "count": counts_s,
                "bin_center": np.round((bin_edges_s[:-1] + bin_edges_s[1:]) / 2, 2),
            })
            sent_hist_df["Sentiment Category"] = [
                "Negative (< -0.05)" if c < -0.05 else ("Positive (> +0.05)" if c > 0.05 else "Neutral (-0.05 to +0.05)")
                for c in sent_hist_df["bin_center"]
            ]

            bars = alt.Chart(sent_hist_df).mark_bar().encode(
                x=alt.X(
                    "bin_center:Q",
                    title="Sentiment score (-1.0 to +1.0)",
                    axis=alt.Axis(labelAngle=0, **AXIS)
                ),
                y=alt.Y("count:Q", title="Accounts", axis=alt.Axis(**AXIS)),
                color=alt.Color(
                    "Sentiment Category:N",
                    scale=alt.Scale(
                        domain=["Negative (< -0.05)", "Neutral (-0.05 to +0.05)", "Positive (> +0.05)"],
                        range=["#d9534f", "#f0ad4e", "#5cb85c"]
                    ),
                    legend=alt.Legend(title="Sentiment category", orient="bottom", **LEGEND)
                ),
                tooltip=["bin_start", "bin_end", "count", "Sentiment Category"]
            )

            rule_neg = alt.Chart(pd.DataFrame({"x": [-0.05]})).mark_rule(color="#d9534f", strokeWidth=2.5, strokeDash=[4, 4]).encode(x="x:Q")
            rule_pos = alt.Chart(pd.DataFrame({"x": [0.05]})).mark_rule(color="#5cb85c", strokeWidth=2.5, strokeDash=[4, 4]).encode(x="x:Q")

            chart = (bars + rule_neg + rule_pos).properties(height=340)
            st.altair_chart(chart, width="stretch")
            st.caption("Dashed lines show the cutoffs. negative is below -0.05 and positive above +0.05")

    with col_nlp:
        # color the rows by domain so its easier to scan
        st.subheader("NLP Root-Cause Issues for Cases")
        if not nlp_themes.empty:
            theme_counts = nlp_themes.copy()
            if "Primary Domain" not in theme_counts.columns:
                theme_counts["Primary Domain"] = "Hardware / Telematics Faults"
            domain_fill = {
                "Billing Disputes": "#f0ad4e",
                "Hardware / Telematics Faults": "#d9534f",
                "Portal Deactivation Stalls": "#0275d8",
                "Support Queue": "#6f42c1",
                "Field Deployment": "#20c997",
            }
            domain_text = {
                "Billing Disputes": "#1a1a1a",
                "Hardware / Telematics Faults": "#ffffff",
                "Portal Deactivation Stalls": "#ffffff",
                "Support Queue": "#ffffff",
                "Field Deployment": "#1a1a1a",
            }

            def color_nlp_rows(row):
                bg = domain_fill.get(row["Primary Domain"], "#6c757d")
                fg = domain_text.get(row["Primary Domain"], "#ffffff")
                return [f"background-color: {bg}; color: {fg}; font-weight: 600"] * len(row)

            styled = (
                theme_counts.style
                .apply(color_nlp_rows, axis=1)
                .set_properties(**{"font-size": "15px", "text-align": "left"})
            )
            st.dataframe(styled, hide_index=True, width="stretch")
            st.caption(
                "Row colors: Yellow = Billing, Red = Hardware, Blue = Portal, Purple = Support Queue, Teal = Field"
            )
        else:
            st.info("This table is missing. Run the notebook export cell again.")

    # same theme table but rolled up into 5 bars
    st.subheader("NLP Root-Cause Category Count")
    if not nlp_themes.empty and "Primary Domain" in nlp_themes.columns:
        freq_col = "Frequency" if "Frequency" in nlp_themes.columns else nlp_themes.columns[1]
        cat_counts = (
            nlp_themes.groupby("Primary Domain", as_index=False)[freq_col]
            .sum()
            .rename(columns={freq_col: "Count", "Primary Domain": "Category"})
        )
        cat_order = ["Billing Disputes", "Hardware / Telematics Faults", "Portal Deactivation Stalls", "Support Queue", "Field Deployment"]
        present = [c for c in cat_order if c in set(cat_counts["Category"])]
        cat_counts["Category"] = pd.Categorical(cat_counts["Category"], categories=present, ordered=True)
        cat_counts = cat_counts.sort_values("Category")

        cat_chart = alt.Chart(cat_counts).mark_bar(size=52, cornerRadiusEnd=3).encode(
            x=alt.X(
                "Category:N",
                sort=present,
                title="Category",
                axis=alt.Axis(labelAngle=0, **AXIS)
            ),
            y=alt.Y("Count:Q", title="Total from table above", axis=alt.Axis(**AXIS)),
            color=alt.Color(
                "Category:N",
                scale=alt.Scale(
                    domain=["Billing Disputes", "Hardware / Telematics Faults", "Portal Deactivation Stalls", "Support Queue", "Field Deployment"],
                    range=["#f0ad4e", "#d9534f", "#0275d8", "#6f42c1", "#20c997"]
                ),
                legend=None
            ),
            tooltip=["Category", "Count"]
        ).properties(height=280)

        st.altair_chart(cat_chart, width="stretch")
        st.caption("Each bar is the sum of matching colors in the table above.")
    else:
        st.info("This chart uses the theme table above, which is missing.")

# page 2 is one account at a time
elif page == "Operator Overview":
    st.title("Operator Overview")
    st.caption("Color is the combined Health Score rank (half logit, half Ops Score).")

    # filter the account list by color first, then pick one account
    st.sidebar.markdown("### Filters")
    filtered = scores.copy()
    if "Risk_Tier" in filtered.columns:
        filtered["Risk_Tier"] = filtered["Risk_Tier"].astype(str).str.strip()

    tier_options = [t for t in TIER_ORDER if "Risk_Tier" in filtered.columns and t in set(filtered["Risk_Tier"].dropna().unique())]
    selected_tiers = st.sidebar.multiselect(
        "Risk tier",
        options=tier_options or TIER_ORDER,
        default=tier_options or TIER_ORDER,
        format_func=lambda t: TIER_LABEL.get(t, t),
    )
    if "Risk_Tier" in filtered.columns:
        if selected_tiers:
            filtered = filtered[filtered["Risk_Tier"].isin(selected_tiers)]
        else:
            filtered = filtered.iloc[0:0]

    link_opt = st.sidebar.selectbox(
        "Account linkage",
        ["All", "Has child sites", "No child sites"],
    )
    if "n_child_sites" in filtered.columns:
        n_kids = pd.to_numeric(filtered["n_child_sites"], errors="coerce").fillna(0)
        if link_opt == "Has child sites":
            filtered = filtered[n_kids > 0]
        elif link_opt == "No child sites":
            filtered = filtered[n_kids == 0]

    st.sidebar.caption(f"{len(filtered):,} account(s) in this list")
    st.sidebar.markdown("### Account")
    all_accounts = sorted(filtered["ent_name"].dropna().unique().tolist()) if "ent_name" in filtered.columns else []
    default_idx = all_accounts.index(default_account) if default_account in all_accounts else 0

    if not all_accounts:
        st.sidebar.warning("No accounts match this filter.")
        selected_ent = None
    else:
        selected_ent = st.sidebar.selectbox(
            "Search or pick an account",
            all_accounts,
            index=default_idx,
        )

    if not selected_ent:
        st.info("Pick at least one risk group in the sidebar.")
    else:
        acct_row = scores[scores["ent_name"] == selected_ent].head(1)

        if not acct_row.empty:
            score_val = float(acct_row["Health_Score"].iloc[0]) if "Health_Score" in acct_row.columns else np.nan
            tier_val = str(acct_row["Risk_Tier"].iloc[0]).strip() if "Risk_Tier" in acct_row.columns else "N/A"
            status_val = acct_row["Account Status"].iloc[0] if "Account Status" in acct_row.columns else "n/a"
            days_this = pd.to_numeric(acct_row["Avg_Days_To_Close"], errors="coerce").iloc[0] if "Avg_Days_To_Close" in acct_row.columns else np.nan
            days_known = pd.to_numeric(scores["Avg_Days_To_Close"], errors="coerce")
            days_med = days_known.median()

            # tiny formatter so n/a shows up instead of crashing on missing values
            def _fmt_metric(col, kind="num"):
                if col not in acct_row.columns:
                    return "n/a"
                v = acct_row[col].iloc[0]
                if pd.isna(v) or v == "":
                    return "n/a"
                if kind == "pct":
                    return f"{float(v) * 100:.1f}%"
                if kind == "flag":
                    return "Yes" if float(v) >= 1 else "No"
                if kind == "int":
                    return f"{int(float(v)):,}"
                if kind == "score":
                    return f"{float(v):.1f}"
                if kind == "text":
                    return str(v)
                try:
                    return f"{float(v):.2f}"
                except (TypeError, ValueError):
                    return str(v)

            # health is half logit / half ops. ops is the average of the 3 scores below.
            st.subheader("Numbers for this account")
            ops_val = pd.to_numeric(acct_row["Ops_Score"], errors="coerce").iloc[0] if "Ops_Score" in acct_row.columns else np.nan
            r1 = st.columns(4)
            r1[0].metric("Health Score", f"{score_val:.1f} / 100" if pd.notna(score_val) else "n/a")
            r1[1].metric("Ops Score", f"{ops_val:.1f} / 100" if pd.notna(ops_val) else "n/a")
            r1[2].metric("Risk status", TIER_LABEL.get(tier_val, tier_val))
            r1[3].metric("Account status", str(status_val))
            st.caption("Health Score = half  logit + half Ops Score. No close time gets a 25 default score on the logit piece. The three scores below make up Ops Score.")

            r2 = st.columns(4)
            r2[0].metric("Avg days to close", _fmt_metric("Avg_Days_To_Close", "score"))
            r2[1].metric("Support queue subscore", _fmt_metric("Support_Queue_Subscore", "score"))
            r2[2].metric("Field deployment subscore", _fmt_metric("Field_Deployment_Subscore", "score"))
            r2[3].metric("VoC sentiment subscore", _fmt_metric("VoC_Sentiment_Subscore", "score"))

            r3 = st.columns(4)
            r3[0].metric("True reopen rate", _fmt_metric("True_Reopen_Rate", "pct"))
            r3[1].metric("Stall ratio", _fmt_metric("Stall_Ratio", "pct"))
            r3[2].metric("Generic triage share", _fmt_metric("Generic_Triage_Share", "pct"))
            r3[3].metric("Unassigned queue ratio", _fmt_metric("Unassigned_Queue_Ratio", "pct"))

            r4 = st.columns(4)
            r4[0].metric("Case volume", _fmt_metric("Total_Case_Volume", "int"))
            r4[1].metric("Repeat contacts", _fmt_metric("Repeat_Contact_Count", "int"))
            r4[2].metric("Call-to-case ratio", _fmt_metric("Call_To_Case_Ratio"))
            r4[3].metric("Partial churn", _fmt_metric("Partial_Churn_Rate", "flag"))

            r5 = st.columns(4)
            r5[0].metric("Mean sentiment", _fmt_metric("Mean_Sentiment_Score"))
            r5[1].metric("Negative comment ratio", _fmt_metric("Negative_Comment_Ratio", "pct"))
            r5[2].metric("Billing dispute flag", _fmt_metric("Has_Billing_Dispute_Flag", "flag"))
            r5[3].metric("Portal deact. stall flag", _fmt_metric("Has_Portal_Deact_Stall_Flag", "flag"))

            n_kids_this = pd.to_numeric(acct_row["n_child_sites"], errors="coerce").iloc[0] if "n_child_sites" in acct_row.columns else np.nan
            if pd.notna(n_kids_this) and float(n_kids_this) > 0:
                hier_val = "Parent"
            else:
                hier_val = "Standalone"

            r6 = st.columns(4)
            r6[0].metric("Hardware fault flag", _fmt_metric("Has_Hardware_Fault_Flag", "flag"))
            r6[1].metric("Hierarchy", hier_val)
            r6[2].metric("Child sites", _fmt_metric("n_child_sites", "int") if "n_child_sites" in acct_row.columns else "n/a")

            def _as_id(x):
                if pd.isna(x) or str(x).strip() in ("", "nan", "None"):
                    return ""
                s = str(x).strip()
                try:
                    return str(int(float(s)))
                except (TypeError, ValueError):
                    return s

            people_this = pd.DataFrame()
            if not people.empty and "ent_num" in acct_row.columns:
                people_this = people.copy()
                people_this["_ent"] = people_this["ent_num"].map(_as_id)
                people_this = people_this[people_this["_ent"] == _as_id(acct_row["ent_num"].iloc[0])]
            n_people = int(people_this["name"].nunique()) if (not people_this.empty and "name" in people_this.columns) else 0
            if n_people == 1:
                one = people_this.iloc[0]
                one_name = str(one.get("name", "")).strip() or "n/a"
                roles = sorted({str(x) for x in people_this["role"].dropna().unique()}) if "role" in people_this.columns else []
                if roles == ["Field engineer"]:
                    p_label = "Field engineer"
                elif roles == ["CS owner"]:
                    p_label = "CS owner"
                else:
                    p_label = "Assigned"
                touches = pd.to_numeric(people_this.get("touches"), errors="coerce").sum() if "touches" in people_this.columns else np.nan
                r7 = st.columns(4)
                r7[0].metric(p_label, one_name)
                r7[1].metric("Touches", f"{int(touches):,}" if pd.notna(touches) else "n/a")

            # explain why this account got its color
            if tier_val == "Red":
                why = "**Red** = combined Health Score is in the lowest 25%."
            elif tier_val == "Yellow":
                why = "**Yellow** = combined Health Score is between the 25th and 50th percentile."
            elif tier_val == "Green":
                why = "**Green** = combined Health Score is in the upper 50%."
            else:
                why = f"**{tier_val}** = how this account ranks on the combined Health Score."

            if tier_val == "Red":
                st.error(why)
            elif tier_val == "Yellow":
                st.warning(why)
            else:
                st.success(why)

            # this strip chart is just days to close vs the other scored accounts
            st.markdown("---")
            st.subheader("Days To Close Risk")
            st.caption("Shorter close times = higher predicted churn in this sample.")
            if pd.isna(days_this):
                st.info("This account has no closed case in Jul-Dec 2024. The logit piece of Health Score is filled with 25.")
            else:
                k1, k2 = st.columns(2)
                k1.metric("This account, days to close", f"{days_this:.1f}")
                k2.metric("Median for accounts with a close time", f"{days_med:.1f}" if pd.notna(days_med) else "n/a")

                days_df = scores[["ent_name", "Risk_Tier", "Avg_Days_To_Close"]].copy()
                days_df["Risk_Tier"] = days_df["Risk_Tier"].astype(str).str.strip()
                days_df["Avg_Days_To_Close"] = pd.to_numeric(days_df["Avg_Days_To_Close"], errors="coerce")
                days_df = days_df.dropna(subset=["Avg_Days_To_Close"])

                red_d = days_df.loc[days_df["Risk_Tier"] == "Red", "Avg_Days_To_Close"]
                yel_d = days_df.loc[days_df["Risk_Tier"] == "Yellow", "Avg_Days_To_Close"]
                grn_d = days_df.loc[days_df["Risk_Tier"] == "Green", "Avg_Days_To_Close"]
                x_max = float(max(days_df["Avg_Days_To_Close"].max(), days_this)) * 1.03
                x_red = float(red_d.max()) if not red_d.empty else 0.0
                x_yel = float(yel_d.max()) if not yel_d.empty else x_red
                if not yel_d.empty and not red_d.empty:
                    x_red = float((red_d.max() + yel_d.min()) / 2)
                if not grn_d.empty and not yel_d.empty:
                    x_yel = float((yel_d.max() + grn_d.min()) / 2)
                x_red = min(max(x_red, 0.5), x_max)
                x_yel = min(max(x_yel, x_red + 0.5), x_max)

                bands = pd.DataFrame([
                    {"x": 0, "x2": x_red, "mid": max(x_red / 2, 1), "band": "High risk", "label": "High"},
                    {"x": x_red, "x2": x_yel, "mid": (x_red + x_yel) / 2, "band": "Moderate", "label": "Mid"},
                    {"x": x_yel, "x2": x_max, "mid": (x_yel + x_max) / 2, "band": "Lower risk", "label": "Low"},
                ])
                shade = alt.Chart(bands).mark_rect(opacity=0.38).encode(
                    x=alt.X("x:Q", title="Days to close", scale=alt.Scale(domain=[0, x_max]), axis=alt.Axis(labelAngle=0, **AXIS)),
                    x2="x2:Q",
                    y=alt.value(28),
                    y2=alt.value(88),
                    color=alt.Color(
                        "band:N",
                        scale=alt.Scale(
                            domain=["High risk", "Moderate", "Lower risk"],
                            range=["#dc3545", "#ffc107", "#28a745"],
                        ),
                        legend=alt.Legend(title=None, orient="bottom", direction="horizontal", columns=3, **LEGEND),
                    ),
                )
                labels = alt.Chart(bands).mark_text(fontSize=14, fontWeight="bold", dy=0).encode(
                    x=alt.X("mid:Q"),
                    y=alt.value(14),
                    text="label:N",
                    color=alt.Color(
                        "band:N",
                        scale=alt.Scale(
                            domain=["High risk", "Moderate", "Lower risk"],
                            range=["#dc3545", "#ffc107", "#28a745"],
                        ),
                        legend=None,
                    ),
                )
                ticks = alt.Chart(days_df).mark_tick(thickness=2, opacity=0.55, size=18).encode(
                    x="Avg_Days_To_Close:Q",
                    y=alt.value(58),
                    color=alt.Color(
                        "Risk_Tier:N",
                        scale=alt.Scale(domain=["Green", "Yellow", "Red"], range=["#1b5e20", "#e65100", "#7f1010"]),
                        legend=None,
                    ),
                )
                here = pd.DataFrame({"Avg_Days_To_Close": [days_this], "note": ["This account"]})
                rule = alt.Chart(here).mark_rule(color="#ffffff", strokeWidth=3).encode(x="Avg_Days_To_Close:Q")
                star = alt.Chart(here).mark_point(size=180, filled=True, color="#ffffff").encode(
                    x="Avg_Days_To_Close:Q",
                    y=alt.value(58),
                    tooltip=["note", alt.Tooltip("Avg_Days_To_Close:Q", format=".1f")],
                )
                st.altair_chart(
                    (shade + labels + ticks + rule + star).properties(
                        height=150,
                        padding={"left": 8, "right": 8, "top": 8, "bottom": 4},
                    ).resolve_scale(color="independent"),
                    width="stretch",
                )
                st.caption("White marker = this account")

            # comments + calls for this account in the jul-dec window
            st.markdown("---")
            st.subheader("Comments and calls")
            st.caption("Comments and calls from Jul-Dec 2024.")

            if timeline.empty or "ent_num" not in acct_row.columns:
                st.info("The comment history file is missing.")
            else:
                ent_num_val = acct_row["ent_num"].iloc[0]
                tl = timeline[timeline["ent_num"] == ent_num_val].copy()
                if "touch_date" in tl.columns:
                    tl["touch_date"] = pd.to_datetime(tl["touch_date"], errors="coerce")
                    tl = tl.dropna(subset=["touch_date"]).sort_values("touch_date", ascending=False)

                if tl.empty:
                    st.info("No comments or calls for this account in Jul-Dec 2024.")
                else:
                    show_tl = tl.copy()
                    rename = {
                        "touch_date": "When",
                        "channel": "Channel",
                        "token": "Contact",
                        "snippet": "What was said",
                    }
                    keep = [c for c in ["touch_date", "channel", "token", "snippet"] if c in show_tl.columns]
                    show_tl = show_tl[keep].rename(columns=rename)
                    if "When" in show_tl.columns:
                        show_tl["When"] = pd.to_datetime(show_tl["When"]).dt.strftime("%Y-%m-%d %H:%M")
                    st.dataframe(show_tl.head(100), hide_index=True, width="stretch")

            st.markdown("---")
            st.subheader("Account linkages")
            st.caption("Billing sites rolled up to this parent. Health is one score for the parent, not a separate score for each child site.")

            if linkage.empty or "ent_num" not in acct_row.columns:
                st.info("No account linkage rows in this csv.")
            else:
                ent_key = _as_id(acct_row["ent_num"].iloc[0])
                link = linkage.copy()
                link["_ent"] = link["ent_num"].map(_as_id)
                link = link[link["_ent"] == ent_key]
                if link.empty:
                    st.info("This account has no billing-site rows in the linkage table.")
                else:
                    show_link = pd.DataFrame({
                        "Billing site": link["account_number"] if "account_number" in link.columns else "",
                        "Token name": link["account_name"] if "account_name" in link.columns else "",
                        "Role": link["role"] if "role" in link.columns else "",
                        "Parent account": link["parent_num"] if "parent_num" in link.columns else "",
                    })
                    st.dataframe(show_link, hide_index=True, width="stretch")

            if n_people > 1:
                st.markdown("---")
                st.subheader("People involved")
                st.caption("Field engineers and CS owners on this account in Jul-Dec 2024.")
                show_p = people_this.copy()
                keep_p = [c for c in ["name", "role", "touches"] if c in show_p.columns]
                show_p = show_p[keep_p].rename(columns={"name": "Person", "role": "Role", "touches": "Touches"})
                if "Touches" in show_p.columns:
                    show_p = show_p.sort_values(["Touches", "Person"], ascending=[False, True])
                else:
                    show_p = show_p.sort_values("Person")
                st.dataframe(show_p, hide_index=True, width="stretch")

# page 3 is just the model tables (what stayed in, what got dropped)
else:
    st.title("Model Summary")
    if "Is_Churned" in scores.columns and "Risk_Tier" in scores.columns:
        # how many accounts that churned between jan-jun 2025 landed in each color
        st.subheader("Did Red catch churn between Jan-Jun 2025?")
        val = (
            scores.assign(Risk_Tier=scores["Risk_Tier"].astype(str).str.strip())
            .groupby("Risk_Tier", dropna=False)
            .agg(accounts=("Risk_Tier", "size"), churns=("Is_Churned", "sum"))
            .reindex(["Red", "Yellow", "Green"])
            .fillna(0)
            .reset_index()
        )
        val["Risk_Tier"] = val["Risk_Tier"].map(lambda x: TIER_LABEL.get(str(x), x))
        val["churns"] = val["churns"].astype(int)
        val["accounts"] = val["accounts"].astype(int)
        val = val.rename(columns={"churns": "Churned between Jan-Jun 2025"})
        st.dataframe(val, hide_index=True, width="stretch")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.subheader("Final model results")
        if not summary.empty:
            st.dataframe(summary, hide_index=True, width="stretch")
        else:
            st.info("The model results table is missing.")

    with col_m2:
        st.subheader("Model Settings")
        if not meta.empty:
            meta_view = meta.copy()
            if "item" in meta_view.columns:
                meta_view = meta_view[~meta_view["item"].isin(
                    ["risk_tier_rule", "vader_status", "claim_type", "n_insufficient_close_time"]
                )]
            st.dataframe(meta_view, hide_index=True, width="stretch")
        else:
            st.info("The model settings table is missing.")

    st.markdown("---")

    st.subheader("Why Variables Were Dropped")
    st.caption("Drop reasons follow the Methodology 2 list. p-value, VIF, and coverage notes are in the detail column.")
    if not disq.empty:
        disq_view = disq.copy()
        if "reason" in disq_view.columns:
            disq_view["reason"] = disq_view["reason"].replace({
                "Statistically Removed": "Not actionable or Lack Statistical Significance",
                "statistically removed": "Not actionable or Lack Statistical Significance",
                "Not actionable": "Not actionable or Lack Statistical Significance",
                "No business value": "Not actionable or Lack Statistical Significance",
                "No Business Value": "Not actionable or Lack Statistical Significance",
                "Does not fit defined health score": "Does not fit the defined health score",
                "Does not fit defined health score / not enough time": "Does not fit the defined health score",
            })
        st.dataframe(disq_view, hide_index=True, width="stretch")
    else:
        st.info("No variables were dropped.")

    st.markdown("---")

    st.subheader("How the Model Changed")
    st.caption("R-squared, AIC, and ROC-AUC after each step of dropping variables.")
    if not iter_df.empty:
        st.dataframe(iter_df, hide_index=True, width="stretch")
    else:
        st.info("The step-by-step model log is missing.")
