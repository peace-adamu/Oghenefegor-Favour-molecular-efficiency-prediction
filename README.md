# Molecular Efficiency Prediction System

This Streamlit app combines the two supplied trained artifacts:

- `best_classification_model.pkl`: `GradientBoostingClassifier`, producing the inhibitor/non-inhibitor verdict and probability.
- `best_regression_model.pkl`: `RandomForestRegressor`, producing the predicted individual efficiency value.

Both artifacts require the same 11 descriptor columns:

`Ionization_Potential`, `Chemical_Hardness`, `Electrophilicity_Index`, `Dipole_Moment`, `Molecular_Weight`, `Num_Aromatic_Ring`, `Num_N`, `Num_O`, `Num_S`, `Num_F`, and `LogP`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

Upload or push the following files to the deployment repository:

- `app.py`
- `requirements.txt`
- `best_classification_model.pkl`
- `best_regression_model.pkl`
- `molecules_with_logp (2).csv`
- `molecules_with_predictions.csv`

Set the app entrypoint to `app.py`. The requirements pin `scikit-learn==1.6.1`, which matches the serialized model metadata and avoids pickle compatibility errors.

## App features

The Prediction page supports manual entry, selection from the 80 reference molecules, and batch CSV upload/download. Additional pages provide the dataset browser, descriptive reference-set model metrics, and ranked top candidates. The explanation chart uses the classifier's built-in global tree feature importances with a directional reference-median proxy; the supplied artifacts do not contain SHAP explainers, so the chart is explicitly not presented as per-observation SHAP values.
