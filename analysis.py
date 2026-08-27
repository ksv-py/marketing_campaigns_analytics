from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ORIGINAL_DATA_URL = "https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign"
REQUIRED_SOURCE_COLUMNS = {"ID", "Year_Birth", "Education", "Marital_Status", "Income", "Response"}


@dataclass
class ModelResult:
    model: Pipeline
    metrics: dict


def load_campaign_data(path: str | None = None) -> pd.DataFrame:
    """Load the original campaign export and derive dashboard-friendly fields."""
    source = path or ORIGINAL_DATA_URL
    source_data = pd.read_csv(source, sep=";")
    missing = REQUIRED_SOURCE_COLUMNS.difference(source_data.columns)
    if missing:
        raise ValueError(f"Campaign export is missing required columns: {', '.join(sorted(missing))}")

    data = source_data.rename(columns={
        "ID": "customer_id", "Education": "education", "Income": "income",
        "Response": "responded", "Recency": "recency_days", "NumWebVisitsMonth": "web_visits_month",
    }).copy()
    data = data.dropna(subset=["income"]).copy()
    data["preferred_channel"] = np.select(
        [data.get("NumWebPurchases", 0) >= data.get("NumStorePurchases", 0), data.get("NumCatalogPurchases", 0) > data.get("NumStorePurchases", 0)],
        ["Web", "Catalog"], default="Store",
    )
    data["segment"] = pd.cut(data["income"], bins=[-np.inf, 35000, 65000, np.inf], labels=["Value conscious", "Growing", "Established"]).astype(str)
    data["campaign"] = np.where(data["AcceptedCmp1"] + data["AcceptedCmp2"] + data["AcceptedCmp3"] + data["AcceptedCmp4"] + data["AcceptedCmp5"] > 0, "Earlier campaign", "Current offer")
    product_columns = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    data["revenue_influenced"] = data[product_columns].sum(axis=1)
    data["prior_purchases"] = data[["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]].sum(axis=1)
    data["web_visits_month"] = data.get("web_visits_month", 0)
    return data


def campaign_summary(data: pd.DataFrame) -> dict:
    return {
        "response_rate": float(data["responded"].mean()) if len(data) else 0,
        "revenue_influenced": float(data["revenue_influenced"].sum()),
        "avg_customer_value": float(data["revenue_influenced"].mean()) if len(data) else 0,
    }


def build_model(data: pd.DataFrame) -> tuple[Pipeline, dict]:
    features = ["education", "preferred_channel", "segment", "income", "recency_days", "web_visits_month", "prior_purchases"]
    categorical = ["education", "preferred_channel", "segment"]
    numerical = ["income", "recency_days", "web_visits_month", "prior_purchases"]
    transformer = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numerical", StandardScaler(), numerical),
    ])
    model = Pipeline([("features", transformer), ("classifier", LogisticRegression(max_iter=500, random_state=24))])
    train, test = train_test_split(data, test_size=0.25, random_state=24, stratify=data["responded"])
    model.fit(train[features], train["responded"])
    probability = model.predict_proba(test[features])[:, 1]
    coefficients = model.named_steps["classifier"].coef_[0]
    names = model.named_steps["features"].get_feature_names_out()
    signals = pd.DataFrame({"signal": names, "direction": np.where(coefficients >= 0, "supports response", "reduces response"), "strength": np.abs(coefficients)})
    return model, {"roc_auc": roc_auc_score(test["responded"], probability), "signals": signals.sort_values("strength", ascending=False)}


if __name__ == "__main__":
    data = build_campaign_data()
    print(data.head())
    print(campaign_summary(data))
