"""
Molecular Efficiency Prediction System
Streamlit UI — Specification v1.0

Models:
  - best_classification_model.pkl  → GradientBoostingClassifier  (High/Not-High)
  - best_regression_model.pkl      → RandomForestRegressor        (IE% continuous)

11 Descriptors (post VIF-pruning from Phase 2):
  Ionization_Potential, Chemical_Hardness, Electrophilicity_Index,
  Dipole_Moment, Molecular_Weight, Num_Aromatic_Ring,
  Num_N, Num_O, Num_S, Num_F, LogP
"""

import io
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DESCRIPTORS = [
    "Ionization_Potential",
    "Chemical_Hardness",
    "Electrophilicity_Index",
    "Dipole_Moment",
    "Molecular_Weight",
    "Num_Aromatic_Ring",
    "Num_N",
    "Num_O",
    "Num_S",
    "Num_F",
    "LogP",
]

DESCRIPTOR_LABELS = {
    "Ionization_Potential":    "Ionization Potential (eV)",
    "Chemical_Hardness":       "Chemical Hardness (eV)",
    "Electrophilicity_Index":  "Electrophilicity Index (eV)",
    "Dipole_Moment":           "Dipole Moment (Debye)",
    "Molecular_Weight":        "Molecular Weight (g/mol)",
    "Num_Aromatic_Ring":       "No. of Aromatic Rings",
    "Num_N":                   "No. of N Atoms",
    "Num_O":                   "No. of O Atoms",
    "Num_S":                   "No. of S Atoms",
    "Num_F":                   "No. of F Atoms",
    "LogP":                    "LogP (Crippen)",
}

# Typical ranges drawn from the dataset for input hints
DESCRIPTOR_RANGES = {
    "Ionization_Potential":   (4.5,  10.5,  6.9),
    "Chemical_Hardness":      (0.5,   4.5,  2.0),
    "Electrophilicity_Index": (1.0,  30.0,  7.5),
    "Dipole_Moment":          (0.0,  25.0,  4.0),
    "Molecular_Weight":       (30.0, 800.0, 150.0),
    "Num_Aromatic_Ring":      (0,     6,    1),
    "Num_N":                  (0,    10,    2),
    "Num_O":                  (0,     8,    1),
    "Num_S":                  (0,     4,    0),
    "Num_F":                  (0,     8,    0),
    "LogP":                   (-5.0, 12.0,  1.5),
}

INTEGER_DESCRIPTORS = {
    "Num_Aromatic_Ring", "Num_N", "Num_O", "Num_S", "Num_F"
}

DATA_PATH  = "molecules_with_predictions.csv"
CLF_PATH   = "best_classification_model.pkl"
REG_PATH   = "best_regression_model.pkl"

GROUP_COLORS = {
    "Organic":         "#4C9BE8",
    "Green Inhibitor": "#4CAF50",
    "Ionic Liquid":    "#FF9800",
    "Inorganic":       "#E84C4C",
}

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Molecular Efficiency Prediction",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #161b22; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #1c2128;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px 16px;
    }

    /* Result card – Inhibitor */
    .verdict-inhibitor {
        background: linear-gradient(135deg, #0d3321 0%, #1a5e38 100%);
        border: 2px solid #2ea44f;
        border-radius: 14px;
        padding: 24px 28px;
        text-align: center;
    }
    /* Result card – Non-Inhibitor */
    .verdict-non-inhibitor {
        background: linear-gradient(135deg, #3b1a1a 0%, #6b2020 100%);
        border: 2px solid #cf222e;
        border-radius: 14px;
        padding: 24px 28px;
        text-align: center;
    }

    /* Section headers */
    .section-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #8b949e;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    /* Tab styling */
    button[data-baseweb="tab"] {
        font-size: 0.95rem;
        font-weight: 600;
    }

    /* Confidence bar label */
    .conf-label {
        font-size: 0.82rem;
        color: #8b949e;
        margin-bottom: 2px;
    }

    hr { border-color: #30363d; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    clf = joblib.load(CLF_PATH)
    reg = joblib.load(REG_PATH)
    return clf, reg


@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: PREDICT FROM A SINGLE ROW
# ─────────────────────────────────────────────────────────────────────────────
def run_prediction(clf, reg, input_dict: dict):
    """Return (label, confidence, ie_predicted, feature_contributions)."""
    X = pd.DataFrame([input_dict])[DESCRIPTORS]

    # Classification
    proba       = clf.predict_proba(X)[0]           # [P(Not-High), P(High)]
    label_idx   = int(np.argmax(proba))
    label       = "Inhibitor" if label_idx == 1 else "Non-Inhibitor"
    confidence  = float(proba[label_idx]) * 100.0

    # Regression
    ie_pred = float(reg.predict(X)[0])

    # Feature contributions via RF feature importances (proxy for SHAP)
    importances = reg.feature_importances_          # shape (11,)
    contribs    = dict(zip(DESCRIPTORS, importances))

    return label, confidence, ie_pred, contribs


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: CONTRIBUTION CHART
# ─────────────────────────────────────────────────────────────────────────────
def contribution_chart(contribs: dict, input_dict: dict):
    """Return a matplotlib Figure of horizontal bar feature contributions."""
    # Sort descending
    sorted_items = sorted(contribs.items(), key=lambda x: x[1], reverse=True)
    names  = [DESCRIPTOR_LABELS[k] for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    colors = ["#2ea44f" if v >= 0 else "#cf222e" for v in values]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.patch.set_facecolor("#1c2128")
    ax.set_facecolor("#1c2128")

    bars = ax.barh(names, values, color=colors, edgecolor="none", height=0.55)

    ax.set_xlabel("Relative Importance", color="#c9d1d9", fontsize=9)
    ax.tick_params(colors="#c9d1d9", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.xaxis.label.set_color("#c9d1d9")
    ax.yaxis.label.set_color("#c9d1d9")

    ax.set_title("Descriptor Contribution to Predicted Efficiency",
                 color="#c9d1d9", fontsize=10, pad=10)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(
            val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left",
            color="#c9d1d9", fontsize=7.5
        )

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧪 Fergor's MEPS")
    st.markdown(
        "<span style='color:#8b949e;font-size:0.82rem'>"
        "Molecular Efficiency Prediction System"
        "</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        """
        <div style='font-size:0.82rem; color:#8b949e; line-height:1.6'>
        <b style='color:#c9d1d9'>Classification model</b><br>
        Gradient Boosting Classifier<br>
        CV Macro F1: 0.5995<br><br>
        <b style='color:#c9d1d9'>Regression model</b><br>
        Random Forest Regressor<br>
        CV R²: 0.0605<br><br>
        <b style='color:#c9d1d9'>Dataset</b><br>
        80 molecules · 4 groups<br>
        11 molecular descriptors
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;color:#484f58'>"
        "Threshold: High ≥ 85 % IE"
        "</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────────────────────────────────────
clf_model, reg_model = load_models()
df_full = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_predict, tab_dataset, tab_performance, tab_candidates = st.tabs([
    "🔬 Prediction",
    "📋 Dataset Browser",
    "📊 Model Performance",
    "🏆 Top Candidates",
])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ═══════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("## Molecular Efficiency Prediction")
    st.markdown(
        "<span style='color:#8b949e'>Input → Run Prediction → View Verdict, "
        "Efficiency Score & Descriptor Contributions</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Input method selector ──────────────────────────────────────────
    input_method = st.radio(
        "Input method",
        ["✏️ Manual Entry", "🔍 Preloaded Molecule", "📁 Batch CSV Upload"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("")

    # ── Shared: initialise descriptor dict ────────────────────────────
    input_values: dict | None = None

    # ── 1. MANUAL ENTRY ───────────────────────────────────────────────
    if input_method == "✏️ Manual Entry":
        st.markdown(
            "<div class='section-header'>Enter 11 Molecular Descriptors</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(3)
        manual = {}
        for i, key in enumerate(DESCRIPTORS):
            lo, hi, default = DESCRIPTOR_RANGES[key]
            with cols[i % 3]:
                if key in INTEGER_DESCRIPTORS:
                    manual[key] = st.number_input(
                        DESCRIPTOR_LABELS[key],
                        min_value=int(lo),
                        max_value=int(hi),
                        value=int(default),
                        step=1,
                        key=f"man_{key}",
                    )
                else:
                    manual[key] = st.number_input(
                        DESCRIPTOR_LABELS[key],
                        min_value=float(lo),
                        max_value=float(hi),
                        value=float(default),
                        step=0.0001,
                        format="%.4f",
                        key=f"man_{key}",
                    )
        input_values = manual

    # ── 2. PRELOADED MOLECULE ─────────────────────────────────────────
    elif input_method == "🔍 Preloaded Molecule":
        st.markdown(
            "<div class='section-header'>Select a molecule from the study dataset</div>",
            unsafe_allow_html=True,
        )
        mol_names = df_full["Molecule"].tolist()
        selected  = st.selectbox(
            "Molecule", mol_names, key="mol_select", label_visibility="collapsed"
        )
        row = df_full[df_full["Molecule"] == selected].iloc[0]

        # Show auto-populated fields
        st.markdown("")
        st.markdown("**Auto-populated descriptor values:**")
        cols = st.columns(3)
        preloaded = {}
        for i, key in enumerate(DESCRIPTORS):
            val = float(row[key])
            preloaded[key] = val
            with cols[i % 3]:
                st.metric(DESCRIPTOR_LABELS[key], f"{val:.4f}")
        input_values = preloaded

    # ── 3. BATCH CSV UPLOAD ───────────────────────────────────────────
    else:
        st.markdown(
            "<div class='section-header'>Upload a CSV with 11 descriptor columns</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"Required columns: `{'`, `'.join(DESCRIPTORS)}`",
            unsafe_allow_html=False,
        )

        uploaded = st.file_uploader(
            "Choose CSV file",
            type=["csv"],
            label_visibility="collapsed",
            key="batch_upload",
        )

        if uploaded:
            try:
                batch_df = pd.read_csv(uploaded)
                missing_cols = [c for c in DESCRIPTORS if c not in batch_df.columns]
                if missing_cols:
                    st.error(f"Missing columns in CSV: {missing_cols}")
                else:
                    st.success(f"Loaded {len(batch_df)} molecules. Running predictions…")

                    # Run predictions for all rows
                    results = []
                    for _, r in batch_df.iterrows():
                        inp = {k: float(r[k]) for k in DESCRIPTORS}
                        lbl, conf, ie_p, _ = run_prediction(clf_model, reg_model, inp)
                        results.append({
                            **{k: r[k] for k in batch_df.columns},
                            "Classification":   lbl,
                            "Confidence_%":     round(conf, 2),
                            "Predicted_IE_pct": round(ie_p, 2),
                        })

                    result_df = pd.DataFrame(results)
                    st.dataframe(result_df, use_container_width=True)

                    # Download button
                    csv_bytes = result_df.to_csv(index=False).encode()
                    st.download_button(
                        "⬇️ Download Results CSV",
                        data=csv_bytes,
                        file_name="batch_predictions.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Error processing file: {e}")

        # No further action for batch mode
        input_values = None

    # ── RUN PREDICTION BUTTON ─────────────────────────────────────────
    if input_values is not None:
        st.markdown("")
        run_btn = st.button(
            "⚡ Run Prediction",
            type="primary",
            use_container_width=False,
        )

        if run_btn:
            label, confidence, ie_pred, contribs = run_prediction(
                clf_model, reg_model, input_values
            )

            st.markdown("---")
            st.markdown("## Results")

            # ── Primary Result Card ────────────────────────────────────
            card_class = (
                "verdict-inhibitor" if label == "Inhibitor"
                else "verdict-non-inhibitor"
            )
            verdict_emoji = "✅" if label == "Inhibitor" else "❌"
            verdict_color = "#2ea44f" if label == "Inhibitor" else "#cf222e"

            st.markdown(
                f"""
                <div class="{card_class}">
                    <div style="font-size:2.6rem; line-height:1">{verdict_emoji}</div>
                    <div style="font-size:1.8rem; font-weight:800;
                                color:{verdict_color}; margin-top:6px">
                        {label}
                    </div>
                    <div style="font-size:0.9rem; color:#8b949e; margin-top:4px">
                        Classification verdict
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("")

            # ── Metrics row ───────────────────────────────────────────
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(
                    "Classification Confidence",
                    f"{confidence:.1f}%",
                    help="Probability from the Gradient Boosting classifier",
                )
            with m2:
                st.metric(
                    "Predicted IE%",
                    f"{ie_pred:.2f}%",
                    help="Continuous efficiency estimate from Random Forest regressor",
                )
            with m3:
                threshold_label = (
                    "High Efficiency (≥ 85%)" if ie_pred >= 85
                    else "Below Threshold (< 85%)"
                )
                st.metric("Efficiency Category", threshold_label)

            # ── Confidence bar ────────────────────────────────────────
            st.markdown("")
            st.markdown(
                "<div class='conf-label'>Classifier confidence breakdown</div>",
                unsafe_allow_html=True,
            )
            proba_all = clf_model.predict_proba(
                pd.DataFrame([input_values])[DESCRIPTORS]
            )[0]
            conf_df = pd.DataFrame({
                "Category":    ["Not-High (< 85%)", "High (≥ 85%)"],
                "Probability": [round(proba_all[0] * 100, 1),
                                round(proba_all[1] * 100, 1)],
            }).set_index("Category")
            st.bar_chart(conf_df, height=140, use_container_width=True)

            # ── Descriptor Contributions ──────────────────────────────
            st.markdown("")
            st.markdown("### Descriptor Contribution Analysis")
            st.markdown(
                "<span style='color:#8b949e;font-size:0.88rem'>"
                "Based on Random Forest feature importances. Higher bars indicate "
                "greater influence on the predicted efficiency score."
                "</span>",
                unsafe_allow_html=True,
            )
            fig = contribution_chart(contribs, input_values)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # ── Contribution table ────────────────────────────────────
            with st.expander("View contribution values as table"):
                sorted_contribs = sorted(
                    contribs.items(), key=lambda x: x[1], reverse=True
                )
                contrib_rows = []
                for k, v in sorted_contribs:
                    direction = (
                        "Strong Positive Contribution" if v >= 0.15
                        else "Positive Contribution" if v >= 0.05
                        else "Minor Contribution"
                    )
                    contrib_rows.append({
                        "Descriptor":   DESCRIPTOR_LABELS[k],
                        "Importance":   round(v, 4),
                        "Contribution": direction,
                    })
                st.dataframe(pd.DataFrame(contrib_rows), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — DATASET BROWSER
# ═══════════════════════════════════════════════════════════════════════
with tab_dataset:
    st.markdown("## Dataset Browser")
    st.markdown(
        "<span style='color:#8b949e'>80 reference molecules from the study dataset</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Filters
    fc1, fc2, fc3 = st.columns([2, 2, 3])
    with fc1:
        groups = ["All"] + sorted(df_full["Group"].unique().tolist())
        sel_group = st.selectbox("Filter by Group", groups, key="db_group")
    with fc2:
        classes = ["All", "High", "Moderate", "Low"]
        sel_class = st.selectbox("Filter by IE Class", classes, key="db_class")
    with fc3:
        search = st.text_input("Search molecule name", placeholder="e.g. Aniline", key="db_search")

    # Apply filters
    filtered = df_full.copy()
    if sel_group != "All":
        filtered = filtered[filtered["Group"] == sel_group]
    if sel_class != "All":
        filtered = filtered[filtered["IE_class"] == sel_class]
    if search.strip():
        filtered = filtered[
            filtered["Molecule"].str.contains(search.strip(), case=False, na=False)
        ]

    st.markdown(f"<span style='color:#8b949e'>Showing {len(filtered)} of 80 molecules</span>",
                unsafe_allow_html=True)
    st.markdown("")

    # Display table
    display_cols = ["ID", "Molecule", "Group", "IE_pct", "IE_class", "IE_pct_predicted"]
    col_rename   = {
        "IE_pct":           "Actual IE%",
        "IE_pct_predicted": "Predicted IE%",
        "IE_class":         "Class",
    }
    st.dataframe(
        filtered[display_cols].rename(columns=col_rename).reset_index(drop=True),
        use_container_width=True,
        height=440,
    )

    # Group distribution chart
    st.markdown("")
    st.markdown("### Group Distribution")
    gcols = st.columns(2)

    with gcols[0]:
        group_counts = df_full["Group"].value_counts()
        fig_g, ax_g = plt.subplots(figsize=(4.5, 3.2))
        fig_g.patch.set_facecolor("#1c2128")
        ax_g.set_facecolor("#1c2128")
        bar_colors = [GROUP_COLORS.get(g, "#8b949e") for g in group_counts.index]
        ax_g.bar(group_counts.index, group_counts.values,
                 color=bar_colors, edgecolor="none", width=0.55)
        ax_g.set_ylabel("Count", color="#c9d1d9", fontsize=9)
        ax_g.tick_params(colors="#c9d1d9", labelsize=8, rotation=15)
        for spine in ax_g.spines.values():
            spine.set_edgecolor("#30363d")
        ax_g.set_title("Molecules by Group", color="#c9d1d9", fontsize=10, pad=8)
        st.pyplot(fig_g, use_container_width=True)
        plt.close(fig_g)

    with gcols[1]:
        class_counts = df_full["IE_class"].value_counts()
        class_colors = {"High": "#2ea44f", "Moderate": "#e3b341", "Low": "#cf222e"}
        fig_c, ax_c = plt.subplots(figsize=(4.5, 3.2))
        fig_c.patch.set_facecolor("#1c2128")
        ax_c.set_facecolor("#1c2128")
        cc = [class_colors.get(c, "#8b949e") for c in class_counts.index]
        ax_c.bar(class_counts.index, class_counts.values,
                 color=cc, edgecolor="none", width=0.45)
        ax_c.set_ylabel("Count", color="#c9d1d9", fontsize=9)
        ax_c.tick_params(colors="#c9d1d9", labelsize=8)
        for spine in ax_c.spines.values():
            spine.set_edgecolor("#30363d")
        ax_c.set_title("Molecules by IE Class", color="#c9d1d9", fontsize=10, pad=8)
        st.pyplot(fig_c, use_container_width=True)
        plt.close(fig_c)

# ═══════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════
with tab_performance:
    st.markdown("## Model Performance")
    st.markdown(
        "<span style='color:#8b949e'>"
        "Comparative metrics for all trained models — Phase 3 results"
        "</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    perf_c1, perf_c2 = st.columns(2)

    # ── Classification ────────────────────────────────────────────────
    with perf_c1:
        st.markdown("### Binary Classification")
        st.markdown(
            "<span style='color:#8b949e;font-size:0.83rem'>"
            "High (IE ≥ 85%) vs Not-High — Gradient Boosting selected by CV Macro F1"
            "</span>",
            unsafe_allow_html=True,
        )
        clf_data = {
            "Model": [
                "Gradient Boosting ⭐",
                "SVC-RBF",
                "Random Forest",
                "Logistic Regression",
                "KNN",
                "XGBoost",
                "Dummy (majority)",
            ],
            "CV Accuracy":  [0.6875, 0.6719, 0.6719, 0.6406, 0.6562, 0.6406, 0.7031],
            "CV Macro F1":  [0.5995, 0.5871, 0.5072, 0.5628, 0.5417, 0.5105, 0.4128],
            "Test Accuracy":[0.5625, 0.6250, 0.7500, 0.6875, 0.5625, 0.6250, 0.6250],
            "Test Macro F1":[0.5152, 0.5595, 0.7091, 0.6537, 0.5152, 0.5000, 0.3846],
            "Test AUC-ROC": [0.5167, 0.5667, 0.5500, 0.6167, 0.4833, 0.5333, 0.5000],
        }
        clf_df = pd.DataFrame(clf_data).set_index("Model")
        st.dataframe(clf_df.style.highlight_max(
            axis=0,
            color="#0d3321",
            subset=["CV Macro F1", "Test Macro F1", "Test AUC-ROC"],
        ), use_container_width=True)

        # Bar chart — CV Macro F1
        fig_clf, ax_clf = plt.subplots(figsize=(5, 3.5))
        fig_clf.patch.set_facecolor("#1c2128")
        ax_clf.set_facecolor("#1c2128")
        models_clf = [m.replace(" ⭐", "") for m in clf_data["Model"]]
        f1_vals    = clf_data["CV Macro F1"]
        bar_cols   = ["#2ea44f" if i == 0 else "#4C9BE8"
                      for i in range(len(models_clf))]
        ax_clf.barh(models_clf, f1_vals, color=bar_cols,
                    edgecolor="none", height=0.55)
        ax_clf.set_xlabel("CV Macro F1", color="#c9d1d9", fontsize=9)
        ax_clf.tick_params(colors="#c9d1d9", labelsize=7.5)
        for spine in ax_clf.spines.values():
            spine.set_edgecolor("#30363d")
        ax_clf.set_title("CV Macro F1 — All Classifiers",
                         color="#c9d1d9", fontsize=10, pad=8)
        ax_clf.axvline(0.5, color="#30363d", linestyle="--", linewidth=0.8)
        plt.tight_layout()
        st.pyplot(fig_clf, use_container_width=True)
        plt.close(fig_clf)

    # ── Regression ────────────────────────────────────────────────────
    with perf_c2:
        st.markdown("### Regression (IE% continuous)")
        st.markdown(
            "<span style='color:#8b949e;font-size:0.83rem'>"
            "Predicts IE% as a continuous value — Random Forest selected by CV R²"
            "</span>",
            unsafe_allow_html=True,
        )
        reg_data = {
            "Model": [
                "Random Forest ⭐",
                "SVR-RBF",
                "KNN",
                "XGBoost",
                "Dummy (Mean)",
                "Gradient Boosting",
                "Lasso",
                "ElasticNet",
                "Ridge",
            ],
            "CV R²":       [ 0.0605,  0.0339,  0.0305, -0.0305, -0.0525, -0.1067, -0.9891, -1.0341, -1.5891],
            "CV RMSE":     [10.8806, 11.0340, 11.0530, 11.3954, 11.5168, 11.8097, 15.8322, 16.0105, 18.0632],
            "CV Spearman": [ 0.3777,  0.3668,  0.3838,  0.3331, -0.2737,  0.2972,  0.3734,  0.3588,  0.3748],
            "Test R²":     [-0.0519, -0.1505, -0.1736, -0.2500, -0.0364, -0.2402, -0.1759, -0.1492, -0.1908],
            "Test RMSE":   [13.9707, 14.6112, 14.7572, 15.2296, 13.8675, 15.1697, 14.7714, 14.6030, 14.8646],
        }
        reg_df = pd.DataFrame(reg_data).set_index("Model")
        st.dataframe(reg_df.style.highlight_max(
            axis=0,
            color="#0d3321",
            subset=["CV R²", "CV Spearman"],
        ), use_container_width=True)

        # Bar chart — CV Spearman
        fig_reg, ax_reg = plt.subplots(figsize=(5, 3.5))
        fig_reg.patch.set_facecolor("#1c2128")
        ax_reg.set_facecolor("#1c2128")
        models_reg = [m.replace(" ⭐", "") for m in reg_data["Model"]]
        rho_vals   = reg_data["CV Spearman"]
        bar_cols_r = ["#2ea44f" if i == 0 else "#4C9BE8"
                      for i in range(len(models_reg))]
        ax_reg.barh(models_reg, rho_vals, color=bar_cols_r,
                    edgecolor="none", height=0.55)
        ax_reg.set_xlabel("CV Spearman ρ", color="#c9d1d9", fontsize=9)
        ax_reg.tick_params(colors="#c9d1d9", labelsize=7.5)
        for spine in ax_reg.spines.values():
            spine.set_edgecolor("#30363d")
        ax_reg.set_title("CV Spearman ρ — All Regressors",
                         color="#c9d1d9", fontsize=10, pad=8)
        ax_reg.axvline(0.3, color="#30363d", linestyle="--", linewidth=0.8)
        plt.tight_layout()
        st.pyplot(fig_reg, use_container_width=True)
        plt.close(fig_reg)

    # ── Model notes ───────────────────────────────────────────────────
    st.markdown("")
    st.markdown("### Model Selection Notes")
    st.info(
        "**Classification:** Gradient Boosting was selected by **CV Macro F1** "
        "(not test-set accuracy) to prevent overfitting to the small 16-molecule test set. "
        "Random Forest showed higher test Macro F1 (0.709) but was not selected.\n\n"
        "**Regression:** All regressors have near-zero or negative R² reflecting the "
        "difficulty of predicting a continuous IE% from 11 descriptors on only 80 molecules. "
        "Random Forest leads on CV R² (0.0605) and is used for the continuous efficiency "
        "estimate and feature-importance analysis.",
        icon="ℹ️",
    )

# ═══════════════════════════════════════════════════════════════════════
# TAB 4 — TOP CANDIDATES
# ═══════════════════════════════════════════════════════════════════════
with tab_candidates:
    st.markdown("## Top Candidate Molecules")
    st.markdown(
        "<span style='color:#8b949e'>"
        "Ranked by predicted Individual Efficiency (IE%) — all 80 study molecules"
        "</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Rank by predicted IE%
    ranked = (
        df_full[["ID", "Molecule", "Group", "IE_pct", "IE_pct_predicted", "IE_class"]]
        .copy()
        .sort_values("IE_pct_predicted", ascending=False)
        .reset_index(drop=True)
    )
    ranked.index += 1                           # 1-based rank
    ranked.index.name = "Rank"

    col_rename_r = {
        "IE_pct":           "Actual IE%",
        "IE_pct_predicted": "Predicted IE%",
        "IE_class":         "Class",
    }
    ranked = ranked.rename(columns=col_rename_r)

    # ── Controls ──────────────────────────────────────────────────────
    tc1, tc2 = st.columns([2, 2])
    with tc1:
        top_n = st.slider("Show top N molecules", 5, 80, 20, key="top_n")
    with tc2:
        group_filter = st.selectbox(
            "Filter by group",
            ["All"] + sorted(df_full["Group"].unique().tolist()),
            key="tc_group",
        )

    display_ranked = ranked.copy()
    if group_filter != "All":
        display_ranked = display_ranked[display_ranked["Group"] == group_filter]
    display_ranked = display_ranked.head(top_n)

    st.dataframe(display_ranked, use_container_width=True, height=420)

    # ── Scatter: actual vs predicted, coloured by group ───────────────
    st.markdown("")
    st.markdown("### Actual vs Predicted IE% — All 80 Molecules")

    fig_s, ax_s = plt.subplots(figsize=(7, 5))
    fig_s.patch.set_facecolor("#1c2128")
    ax_s.set_facecolor("#1c2128")

    legend_handles = []
    for grp, grp_df in df_full.groupby("Group"):
        color = GROUP_COLORS.get(grp, "#8b949e")
        ax_s.scatter(
            grp_df["IE_pct"], grp_df["IE_pct_predicted"],
            c=color, s=55, alpha=0.85, edgecolors="none", label=grp
        )
        legend_handles.append(
            mpatches.Patch(color=color, label=grp)
        )

    # Perfect-prediction line
    all_vals = pd.concat([df_full["IE_pct"], df_full["IE_pct_predicted"]])
    lo_v, hi_v = all_vals.min() - 2, all_vals.max() + 2
    ax_s.plot([lo_v, hi_v], [lo_v, hi_v],
              color="#e3b341", linestyle="--", linewidth=1.1,
              label="Perfect prediction")

    ax_s.set_xlabel("Actual IE%", color="#c9d1d9", fontsize=10)
    ax_s.set_ylabel("Predicted IE%", color="#c9d1d9", fontsize=10)
    ax_s.tick_params(colors="#c9d1d9", labelsize=8.5)
    for spine in ax_s.spines.values():
        spine.set_edgecolor("#30363d")
    ax_s.set_title("Actual vs Predicted IE% (Random Forest, all 80 molecules)",
                   color="#c9d1d9", fontsize=10, pad=10)
    legend = ax_s.legend(
        handles=legend_handles + [
            plt.Line2D([0], [0], color="#e3b341", linestyle="--",
                       linewidth=1.1, label="Perfect prediction")
        ],
        fontsize=8, facecolor="#1c2128", edgecolor="#30363d",
        labelcolor="#c9d1d9",
    )
    plt.tight_layout()
    st.pyplot(fig_s, use_container_width=True)
    plt.close(fig_s)

    # ── Download ranked list ──────────────────────────────────────────
    st.markdown("")
    csv_ranked = ranked.to_csv().encode()
    st.download_button(
        "⬇️ Download Full Ranked List (CSV)",
        data=csv_ranked,
        file_name="top_candidates.csv",
        mime="text/csv",
    )
