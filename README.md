# SpendSmart-E-Commerce-Super-Category-Tagging-Next-Purchase-Prediction-with-Random-Forest-and-LSTM

Marketplaces add thousands of items/orders daily. Two recurring problems cost time and revenue:

Catalog hygiene: items/orders lack reliable super-category tags - weaker search facets, messy analytics, costly manual QA.

Next-best-action: customers usually repeat categories, but the rare category switch is the profitable cross-sell moment most heuristics miss.

SpendSmart tackles both with one pipeline on real retail (Kaggle Olist–style) data:
 
1.Auto-tag orders into super-categories from tabular signals.

2. Predict a customer’s next super-category from their purchase history, highlighting switch events.

# Project overview (tasks & data)

Data: Kaggle Olist-style retail tables (orders, items, products, payments, customers) joined into one modeling table; super_category derived from product metadata.

Task 1 (Tabular): Predict the super-category of each order from price, freight, weight/volume, payment, and calendar features.

Task 2 (Sequential): Predict a customer’s next super-category from their chronological category sequence.

# What this project demonstrates

End-to-end ML on real data: joining tables, feature engineering, class-imbalance handling, fair splits/metrics.

Strong tabular modeling with comparative baselines (RF, LightGBM, HGB) and clear evaluation (Accuracy + Macro-F1).

Sequence modeling with a 2-layer LSTM benchmarked against a strong repeat-last baseline.

Trustworthy reporting: confusion matrices and targeted sanity checks (where the LSTM actually adds value—on switches).

# Tech stack

Python, pandas, NumPy

scikit-learn (RandomForest, HistGradientBoosting, pipelines, metrics)

LightGBM

TensorFlow/Keras (Embedding + LSTM)

Matplotlib/Seaborn for plots

# Workflow

**1. Data prep & joins**: assemble modeling table from Olist-style CSVs.

**2. Feature engineering (Task 1):**

* log1p_{price, payment_value, product_weight_g, product_volume_cm3} (tame heavy tails)

* freight_ratio = freight_value / price (clipped to [0,5])

* Calendar: month, weekday

* Payment multi-hot flags: pay_credit_card, pay_boleto, pay_voucher, pay_debit_card

* Replace ±∞ NaN; median imputation in pipelines

**3. Imbalance handling:** inverse-frequency class weights (or balanced_subsample for RF).

**4. Modeling (Task 1):** RF (headline), LGBM (tie), HGB (fast ablation). Metrics: Accuracy + Macro-F1 on a stratified split.

**5. Sequences (Task 2):**

* Build per-customer chronological category lists; keep sequences with ≥3 purchases

* Split by customer (no leakage)

* Baseline = repeat-last

* Model = Embedding - LSTM(128) - Dropout(0.3) - LSTM(128) - Dense softmax; class weights; early stopping

* Metrics: Top-1 / Top-3; plus a non-persistent (switch) slice

* Sanity checks: persistence rate, switch slice lift, confusion matrices.

# Key results 

**Task 1 - Super-Category Tagging**


| Model                                 |  Accuracy |  Macro-F1 | Notes         |
| ------------------------------------- | :-------: | :-------: | ------------- |
| **RandomForest (balanced_subsample)** | **0.813** | **~0.78** | Headline      |
| LightGBM (class-weighted)             |   0.812   |  ~RF tie  | Competitive   |
| HistGradientBoosting (weighted)       |   ~0.60   |   ~0.54   | Fast ablation |


**Task 2 — Next-Purchase Category**

* Baseline (repeat-last): Top-1 0.968

* LSTM (2-layer): Top-1 0.972, Top-3 0.985

* Switch steps (~3% of cases): baseline 0.00 - LSTM 0.24 (Top-1)

Overall behavior is persistent, but LSTM adds meaningful lift on switch events—the moments that matter for cross-sell.


# Skills demonstrated

- Data wrangling & joins on real marketplace data

- Feature engineering for tabular ML (log transforms, ratios, calendar, multi-hot)

- Class-imbalance handling and Macro-F1 reporting

- Comparative modeling (RF/LGBM/HGB) with pipelines

- Sequence modeling (Embedding + LSTM) and fair customer-level evaluation

- Error analysis (confusions, switch-slice sanity checks)

# Dataset
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?utm_source=chatgpt.com














