# NBA – Next Best Action (Streamlit App)

LSTM-based product recommendation for Santander customers.  
Predicts which of 10 products a customer is likely to add next month, given 16 months of history.

## Setup

1. Place these 3 model files in the same folder as `app.py`:
   - `nba_model_final.pt`
   - `nba_scaler.pkl`
   - `nba_feature_cols.json`

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run:
```bash
streamlit run app.py
```

## Notes
- scikit-learn version must be **1.6.1** (matches the version used to save the scaler)
- Model runs on CPU — no GPU needed for inference
- Input: 16 months of product ownership + customer demographics
- Output: probability scores for 10 target products in month 17
