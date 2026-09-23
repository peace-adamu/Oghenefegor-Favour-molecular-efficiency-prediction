from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


# -----------------------------
# Configuration and data loading
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
CLASSIFIER_PATH = BASE_DIR / "best_classification_model.pkl"
REGRESSOR_PATH = BASE_DIR / "best_regression_model.pkl"
DATA_PATH = BASE_DIR / "molecules_with_logp (2).csv"
PREDICTIONS_PATH = BASE_DIR / "molecules_with_predictions.csv"

FEATURES = [
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

FEATURE_LABELS = {
    "Ionization_Potential": "Ionization potential",
    "Chemical_Hardness": "Chemical hardness",
    "Electrophilicity_Index": "Electrophilicity index",
    "Dipole_Moment": "Dipole moment",
    "Molecular_Weight": "Molecular weight",
    "Num_Aromatic_Ring": "Aromatic rings",
    "Num_N": "Nitrogen atoms",
    "Num_O": "Oxygen atoms",
    "Num_S": "Sulfur atoms",
    "Num_F": "Fluorine atoms",
    "LogP": "LogP",
}

DISPLAY_COLUMNS = ["Molecule", "Group", "IE_pct", "IE_class"]

st.set_page_config(
    page_title="Molecular Efficiency Prediction",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading trained models...")
def load_models():
    """Load the exact artifacts supplied with the project.

    These pickles were trained with scikit-learn 1.6.1; that version is pinned
    in requirements.txt for reproducible deployment.
    """
    if not CLASSIFIER_PATH.exists() or not REGRESSOR_PATH.exists():
        raise FileNotFoundError("Both model files must be stored beside app.py.")
    return joblib.load(CLASSIFIER_PATH), joblib.load(REGRESSOR_PATH)


@st.cache_data(show_spinner=False)
def load_reference_data():
    if PREDICTIONS_PATH.exists():
        return pd.read_csv(PREDICTIONS_PATH)
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def load_raw_data():
    return pd.read_csv(DATA_PATH)


def safe_load():
    try:
        return load_models()
    except Exception as exc:
        st.error("The trained models could not be loaded.")
        st.exception(exc)
        st.info("Confirm that both .pkl files are beside app.py and that scikit-learn 1.6.1 is installed.")
        st.stop()


classifier, regressor = safe_load()
reference_df = load_reference_data()
raw_df = load_raw_data()

# Use the model's own feature contract when available, while validating it.
model_features = list(getattr(classifier, "feature_names_in_", FEATURES))
if model_features != FEATURES:
    st.warning(f"The classifier feature order differs from the expected project schema: {model_features}")
FEATURES = model_features


# -----------------------------
# Prediction helpers
# -----------------------------
def make_feature_frame(values: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        missing = [feature for feature in FEATURES if feature not in values.columns]
        if missing:
            raise ValueError("Missing required descriptor columns: " + ", ".join(missing))
        frame = values[FEATURES].copy()
    else:
        frame = pd.DataFrame([{feature: values[feature] for feature in FEATURES}])
    for feature in FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    if frame.isna().any().any():
        missing = frame.columns[frame.isna().any()].tolist()
        raise ValueError("All descriptor values must be numeric. Invalid or missing: " + ", ".join(missing))
    return frame


def predict_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return predictions for one or many rows."""
    classification = classifier.predict(frame).astype(int)
    probabilities = classifier.predict_proba(frame)
    class_to_probability = {int(label): probabilities[:, i] for i, label in enumerate(classifier.classes_)}
    confidence = np.array([class_to_probability[label][i] for i, label in enumerate(classification)])
    efficiency = regressor.predict(frame)
    result = pd.DataFrame(
        {
            "Predicted_Label": np.where(classification == 1, "Inhibitor", "Non-Inhibitor"),
            "Classification_Confidence_pct": confidence * 100,
            "Predicted_Individual_Efficiency": efficiency,
            "Predicted_Binary_Class": classification,
        },
        index=frame.index,
    )
    return result


def importance_frame(values: pd.DataFrame) -> pd.DataFrame:
    """Build a transparent, input-aware feature-importance view.

    The supplied artifacts expose tree feature_importances_, not SHAP values.
    We therefore show global importance and a directional proxy based on whether
    the current descriptor is above or below the reference median. It is not a
    causal or exact per-observation SHAP decomposition.
    """
    reference = raw_df[FEATURES].apply(pd.to_numeric, errors="coerce")
    medians = reference.median()
    importance = np.asarray(getattr(classifier, "feature_importances_", np.zeros(len(FEATURES))))
    current = values.iloc[0]
    directions = np.where(current.to_numpy() >= medians.to_numpy(), 1.0, -1.0)
    signed = importance * directions
    output = pd.DataFrame(
        {
            "Descriptor": [FEATURE_LABELS.get(feature, feature) for feature in FEATURES],
            "Feature": FEATURES,
            "Importance": importance,
            "Directional contribution": signed,
        }
    ).sort_values("Directional contribution")
    return output


def render_prediction_card(result: pd.Series):
    label = result["Predicted_Label"]
    confidence = float(result["Classification_Confidence_pct"])
    efficiency = float(result["Predicted_Individual_Efficiency"])
    color = "#198754" if label == "Inhibitor" else "#6c757d"
    st.markdown(
        f"""
        <div style='border:1px solid #d7dde5;border-left:8px solid {color};border-radius:10px;padding:1.1rem 1.25rem;background:#ffffff;margin:0.4rem 0 1rem 0;'>
          <div style='font-size:0.82rem;color:#667085;text-transform:uppercase;letter-spacing:.08em;'>Classification verdict</div>
          <div style='font-size:2.0rem;font-weight:700;color:{color};margin:.15rem 0 .4rem 0;'>{label}</div>
          <div style='display:flex;gap:2.5rem;flex-wrap:wrap;'>
            <div><span style='color:#667085;'>Classification confidence</span><br><b>{confidence:.2f}%</b></div>
            <div><span style='color:#667085;'>Predicted individual efficiency</span><br><b>{efficiency:.2f}%</b></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_explanation(frame: pd.DataFrame):
    st.subheader("Descriptor contribution analysis")
    st.caption("Directional feature-importance proxy from the Gradient Boosting classifier. Positive bars indicate values above the reference median; negative bars indicate values below it. This is not a per-row SHAP decomposition.")
    explanation = importance_frame(frame)
    fig = px.bar(
        explanation,
        x="Directional contribution",
        y="Descriptor",
        orientation="h",
        color="Directional contribution",
        color_continuous_scale=["#c2415b", "#d9dee7", "#198754"],
        labels={"Directional contribution": "Signed importance proxy", "Descriptor": ""},
        hover_data={"Feature": True, "Importance": ":.4f"},
    )
    fig.update_layout(height=440, coloraxis_showscale=False, margin=dict(l=10, r=10, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Sidebar and page navigation
# -----------------------------
st.sidebar.title("Molecular Efficiency")
st.sidebar.caption("Gradient Boosting classification + Random Forest regression")
page = st.sidebar.radio("Navigate", ["Prediction", "Dataset Browser", "Model Performance", "Top Candidates"])
st.sidebar.divider()
st.sidebar.markdown("**Model artifacts**")
st.sidebar.write("• GradientBoostingClassifier")
st.sidebar.write("• RandomForestRegressor")
st.sidebar.caption("Both models use the same 11 descriptors.")


# -----------------------------
# Prediction page
# -----------------------------
if page == "Prediction":
    st.title("Molecular Efficiency Prediction")
    st.write("Enter molecular descriptors, choose a reference molecule, or upload a batch CSV to obtain an inhibitor verdict, confidence, efficiency estimate, and descriptor explanation.")

    input_mode = st.radio("Input method", ["Manual Entry", "Preloaded Molecule", "Batch CSV Upload"], horizontal=True)

    if input_mode == "Batch CSV Upload":
        st.subheader("Batch CSV prediction")
        st.write("Upload a CSV containing the 11 required descriptor columns. Additional columns such as Molecule or ID are preserved in the download.")
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded is not None:
            batch = pd.read_csv(uploaded)
            st.write(f"Loaded **{len(batch):,}** rows and **{len(batch.columns):,}** columns.")
            missing = [feature for feature in FEATURES if feature not in batch.columns]
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
                st.info("Download the template below to see the exact feature names.")
            else:
                st.dataframe(batch.head(10), use_container_width=True, hide_index=True)
                if st.button("Run batch prediction", type="primary"):
                    try:
                        frame = make_feature_frame(batch)
                        predictions = predict_frame(frame)
                        export = pd.concat([batch.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)
                        st.success(f"Predicted {len(export):,} molecules.")
                        st.dataframe(export, use_container_width=True, hide_index=True)
                        st.download_button("Download prediction results", export.to_csv(index=False).encode("utf-8"), "molecular_predictions.csv", "text/csv")
                    except Exception as exc:
                        st.error(str(exc))
        template = raw_df[FEATURES].head(1).to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV template", template, "molecular_descriptor_template.csv", "text/csv")

    else:
        selected_row = None
        if input_mode == "Preloaded Molecule":
            options = raw_df["Molecule"].astype(str).tolist()
            selected = st.selectbox("Select one of the 80 reference molecules", options)
            selected_row = raw_df.loc[raw_df["Molecule"].astype(str) == selected].iloc[0]
            st.caption(f"Reference molecule: {selected} · Group: {selected_row.get('Group', '—')}")

        with st.form("prediction_form"):
            st.subheader("Molecular descriptors")
            cols = st.columns(3)
            values = {}
            for i, feature in enumerate(FEATURES):
                default = None if selected_row is None else float(selected_row[feature])
                with cols[i % 3]:
                    values[feature] = st.number_input(
                        FEATURE_LABELS.get(feature, feature),
                        value=default,
                        format="%.6f",
                        help=f"Model feature: {feature}",
                    )
            submitted = st.form_submit_button("Run prediction", type="primary", use_container_width=True)

        if submitted:
            if any(value is None for value in values.values()):
                st.error("All 11 descriptor fields are required.")
            else:
                try:
                    frame = make_feature_frame(values)
                    result = predict_frame(frame).iloc[0]
                    st.divider()
                    render_prediction_card(result)
                    render_explanation(frame)
                except Exception as exc:
                    st.error(str(exc))


# -----------------------------
# Dataset browser
# -----------------------------
elif page == "Dataset Browser":
    st.title("Dataset Browser")
    st.write("The 80 reference molecules from the study dataset.")
    browser = reference_df.copy()
    browser["Individual Efficiency Value"] = browser.get("IE_pct", np.nan)
    browser["Classification Label"] = np.where(browser.get("IE_binary", 0).astype(int) == 1, "Inhibitor", "Non-Inhibitor")
    group_filter = st.multiselect("Filter by group", sorted(browser["Group"].dropna().unique()), default=sorted(browser["Group"].dropna().unique()))
    browser = browser[browser["Group"].isin(group_filter)]
    view = browser[["Molecule", "Group", "Individual Efficiency Value", "Classification Label"]].copy()
    view = view.rename(columns={"Molecule": "Molecule Name"})
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(view)} of {len(reference_df)} reference molecules.")


# -----------------------------
# Model performance
# -----------------------------
elif page == "Model Performance":
    st.title("Model Performance")
    st.warning("Metrics below are calculated on the supplied 80-row reference dataset. They are descriptive reference-set metrics, not an independent holdout evaluation.")
    try:
        eval_frame = make_feature_frame(reference_df)
        predictions = predict_frame(eval_frame)
        y_class = pd.to_numeric(reference_df["IE_binary"], errors="coerce").astype(int).to_numpy()
        y_eff = pd.to_numeric(reference_df["IE_pct"], errors="coerce").to_numpy()
        y_pred_class = predictions["Predicted_Binary_Class"].to_numpy()
        y_proba = classifier.predict_proba(eval_frame)[:, list(classifier.classes_).index(1)]
        y_pred_eff = predictions["Predicted_Individual_Efficiency"].to_numpy()
        metrics = pd.DataFrame(
            [
                {"Model": "Gradient Boosting classifier", "Accuracy": accuracy_score(y_class, y_pred_class), "F1 Score": f1_score(y_class, y_pred_class, zero_division=0), "AUC-ROC": roc_auc_score(y_class, y_proba), "MAE": np.nan, "R²": np.nan},
                {"Model": "Random Forest regressor", "Accuracy": np.nan, "F1 Score": np.nan, "AUC-ROC": np.nan, "MAE": mean_absolute_error(y_eff, y_pred_eff), "R²": r2_score(y_eff, y_pred_eff)},
            ]
        )
        st.dataframe(metrics.style.format({"Accuracy": "{:.3f}", "F1 Score": "{:.3f}", "AUC-ROC": "{:.3f}", "MAE": "{:.3f}", "R²": "{:.3f}"}, na_rep="—"), use_container_width=True, hide_index=True)
        st.subheader("Reference-set classification metrics")
        metric_plot = metrics.iloc[[0]][["Accuracy", "F1 Score", "AUC-ROC"]].melt(var_name="Metric", value_name="Score")
        fig = px.bar(metric_plot, x="Metric", y="Score", range_y=[0, 1], color="Metric", text_auto=".3f")
        fig.update_layout(showlegend=False, height=350, margin=dict(l=10, r=10, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.error("Could not calculate metrics from the supplied reference data.")
        st.exception(exc)


# -----------------------------
# Top candidates
# -----------------------------
else:
    st.title("Top Candidates")
    st.write("Reference molecules ranked by the Random Forest predicted individual efficiency.")
    candidates = reference_df.copy()
    if "IE_pct_predicted" not in candidates.columns:
        frame = make_feature_frame(candidates)
        candidates = pd.concat([candidates.reset_index(drop=True), predict_frame(frame).reset_index(drop=True)], axis=1)
        efficiency_column = "Predicted_Individual_Efficiency"
    else:
        efficiency_column = "IE_pct_predicted"
    candidates["Predicted classification"] = np.where(candidates.get("IE_binary", 0).astype(int) == 1, "Inhibitor", "Non-Inhibitor")
    candidates = candidates.sort_values(efficiency_column, ascending=False).reset_index(drop=True)
    candidates.insert(0, "Rank", np.arange(1, len(candidates) + 1))
    top_n = st.slider("Number of candidates to display", min_value=5, max_value=min(80, len(candidates)), value=20)
    top_view = candidates.head(top_n)[["Rank", "Molecule", "Group", efficiency_column, "Predicted classification"]].rename(columns={efficiency_column: "Predicted efficiency"})
    st.dataframe(top_view.style.format({"Predicted efficiency": "{:.2f}%"}), use_container_width=True, hide_index=True)
    chart = px.bar(top_view.sort_values("Predicted efficiency"), x="Predicted efficiency", y="Molecule", orientation="h", color="Group", hover_data=["Rank", "Predicted classification"])
    chart.update_layout(height=max(420, top_n * 25), margin=dict(l=10, r=10, t=20, b=20))
    st.plotly_chart(chart, use_container_width=True)
    st.download_button("Download ranked candidates", candidates.to_csv(index=False).encode("utf-8"), "ranked_molecular_candidates.csv", "text/csv")
