import pandas as pd
from sklearn.linear_model import LogisticRegression

from churn.train import fit_pipeline, load


def tiny_frame():
    rows = []
    for i in range(60):
        rows.append({"tenure": i % 12, "MonthlyCharges": 20 + i, "TotalCharges": 100.0 + i,
                     "SeniorCitizen": i % 2, "Contract": "Month-to-month" if i % 3 else "Two year"})
    X = pd.DataFrame(rows)
    y = pd.Series([(r["tenure"] < 4 and r["Contract"] == "Month-to-month") for r in rows], dtype=int)
    return X, y


def test_pipeline_fits_and_predicts():
    X, y = tiny_frame()
    pipe = fit_pipeline(X, LogisticRegression(max_iter=1000)).fit(X, y)
    proba = pipe.predict_proba(X.head(3))
    assert proba.shape == (3, 2)


def test_load_handles_totalcharges_blanks(tmp_path):
    csv = tmp_path / "toy.csv"
    csv.write_text(
        "customerID,tenure,MonthlyCharges,TotalCharges,Contract,Churn\n"
        "a,1,10,10,Month-to-month,Yes\n"
        "b,2,20, ,Two year,No\n"
    )
    X, y = load(str(csv))
    assert X.loc[1, "TotalCharges"] == 0
    assert y.tolist() == [1, 0]
