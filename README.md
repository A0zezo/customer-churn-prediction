# Customer Churn Prediction

Train and serve a churn classifier on the public Telco Customer Churn dataset (ships as a 1,600-customer stratified sample; full dataset: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv): scikit-learn pipelines with proper preprocessing, honest held-out metrics, model selection by ROC AUC, and a FastAPI prediction endpoint.

## Results (real, held-out 20% test split, seed 42)

```
train=5634 test=1409 churn_rate=0.265
logistic_regression: accuracy=0.7875  precision=0.6104  recall=0.5529  f1=0.5802  roc_auc=0.8352
random_forest:       accuracy=0.7906  precision=0.6216  recall=0.5412  f1=0.5786  roc_auc=0.805
saved churn-model.joblib (logistic_regression, roc_auc=0.8352)
```

Logistic regression wins on ROC AUC here and is the model that gets saved and served. The class balance (26.5% churn) is why accuracy alone would be misleading: a model that never predicts churn already scores 73.5%.

## Usage

```bash
pip install -r requirements.txt
python -m churn.train            # trains, evaluates, saves churn-model.joblib
uvicorn churn.api:app            # serves POST /api/predict
```

```bash
curl -X POST http://localhost:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":1,"PhoneService":"No","MultipleLines":"No phone service","InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":29.85,"TotalCharges":29.85}'
```

Real response for that customer (1 month tenure, month-to-month contract):

```json
{"churn_probability": 0.6137, "churn": true, "model": "logistic_regression"}
```

## Design notes

- One `ColumnTransformer` pipeline does scaling (numeric) + one-hot encoding (categorical), so preprocessing is fit on the training split only - no leakage - and travels with the saved model.
- `TotalCharges` has blank strings in the raw CSV; they are coerced to 0 (those are 0-tenure customers).
- `customerID` is dropped; `handle_unknown="ignore"` keeps the API robust to categories unseen in training.

## Data

`data/telco-churn.csv` is the widely used IBM Telco sample (also on Kaggle as "WA_Fn-UseC_-Telco-Customer-Churn"). 21 columns: demographics, services, contract, charges, and the `Churn` label.

## Honest limitations

- Two baseline models only; no hyperparameter search, no gradient boosting (XGBoost/LightGBM typically adds ~1-2 AUC points on this dataset).
- No calibration: probabilities rank customers well but are not calibrated percentages.
- The dataset is synthetic-ish vendor sample data, not production traffic.

## Tests

```bash
pytest -q
```
