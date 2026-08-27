# Campaign Lens

Campaign Lens is an original marketing response analysis project inspired by the business questions in the reference repository. It is written as a small analyst tool rather than a copied notebook: it uses the original public Marketing Campaign customer export, the narrative is decision-oriented, and the model is treated as a baseline for prioritisation rather than proof of causality.

## What it answers

- Which campaigns and channels earn attention that turns into responses?
- Which customer segments are worth prioritising?
- How much revenue is associated with the responding audience?
- Which signals can support a first-pass response propensity score?

## Run it

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Download the original `marketing_campaign.csv` file from the [Kaggle Marketing Campaign dataset](https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign), then place it at `data/marketing_campaign.csv`. The loader accepts the source columns (`ID`, `Year_Birth`, `Education`, `Income`, `Response`, and the campaign/product fields) and derives the dashboard fields in `analysis.py`.

## Notes on interpretation

The dataset is synthetic and the revenue field is campaign-associated, not a verified causal lift measure. Before using this workflow with real customers, add consent checks, a time-based evaluation split, calibration, and an experiment design to measure incremental impact.
