# King County Valuations

Estimating the sale price of a house in King County, Washington from its physical characteristics and location, and serving that estimate through an interactive Streamlit app.

**Machine Learning for Developers (CAI2C08), Individual Project**
Diploma in Applied Artificial Intelligence, School of Informatics & IT, Temasek Polytechnic

Student: Lau Yan Ler Jovan (2501328E)

**Live app:** https://king-county-valuations-mc63jjbju8ztmtsfdcnjdq.streamlit.app/

## Problem Statement

House pricing affects many parties in a property transaction. Sellers want to list at a price that
attracts buyers without leaving money on the table. Buyers want to know whether an asking price is
fair before committing to an offer. Real estate agents sit between both groups and are often
required to justify a listing price quickly, whether to a seller pushing for a higher number or a
buyer questioning whether a price is justified.

Pricing by experience alone is slow and hard to defend, especially for agents managing many
listings or without years of local pricing intuition. A data-driven price estimate gives all three
parties a fast, evidence-based reference point, most directly useful to the agent, who can use it
to support pricing conversations with clients on both sides of a deal.

This is a **regression** problem, because the target variable (`price`) is a numerical value
rather than a category.

## Data

King County House Sales dataset from Kaggle
https://www.kaggle.com/datasets/harlfoxem/housesalesprediction/data
21,613 property sales in King County, Washington, recorded between May 2014 and May 2015.

| | |
|---|---|
| Rows | 21,613 (one row = one recorded sale) |
| Features | 21 raw columns, 18 after dropping `id` and `date`, 20 after feature engineering, **88 after one-hot encoding** |
| Target | `price`, continuous, mean roughly $540,000 |
| Missing values | None. `isnull().sum()` returns 0 for every column |
| Zipcodes | 70 distinct, one-hot encoded to 69 dummy columns |

### Three things about this data that shaped everything after

**`zipcode` is a location label stored as `int64`.** Left as a number, a model is told that 98004
is "greater than" 98003 by exactly one unit, which is meaningless for a neighbourhood identifier.
This hurts Linear Regression most severely, since it must fit a single straight-line coefficient
across all 70 locations. It is the single largest reason Linear Regression finished last at
baseline (R² 0.7103 against Random Forest's 0.8837).

**`yr_renovated` uses 0 as a sentinel for "never renovated".** Read literally, an un-renovated
house is dated to the year 0, thousands of years before the oldest real building in the dataset.
Converting this to a yes/no flag fixes a genuine data defect regardless of whether it moves the
metric.

**Bathrooms have fractional values.** I was initially confused by values such as 2.25. US property
listings count bathrooms in fractions: 1.0 is a full bath (toilet, sink, bath/shower), 0.75 is a
three-quarter bath (toilet, sink, shower, no tub), and 0.5 is a powder room (toilet and sink only).
So "2.25 bathrooms" means two full baths plus a powder room. This is a legitimate continuous
measure, not dirty data, so it is left as-is.

## Results

The deployed model is a **tuned Random Forest** (`n_estimators=300`, `min_samples_split=2`,
`max_depth=20`), trained on 70% of the cleaned data and scored on a held-out 30%.

| Metric | Score |
|---|---|
| **MAE** | **$68,933** |
| RMSE | $127,305 |
| R² | 0.8861 |

An MAE of $68,933 against a mean sale price of roughly $540,000 means a typical estimate lands
within about **13%** of the true price. That is close enough to anchor a pricing conversation or
sanity-check a seller's expectations, but not precise enough to set a final listing price without
an agent's local judgement on the individual property.

### Why MAE, and why not F1

This is a regression problem, so classification metrics such as accuracy or F1 do not apply. There
are no classes to get right or wrong, only a distance between predicted and actual price. Three
complementary metrics are reported:

* **MAE** is the headline. It is in dollars, so it answers the question the agent actually asks:
  how far off is this estimate likely to be? It also weights every dollar of error equally, which
  suits a tool used mostly on typical homes rather than outliers.
* **RMSE** squares errors before averaging, so it penalises large misses far more heavily.
  Reported next to MAE it works as a diagnostic: RMSE ($127,305) is roughly **1.8× MAE**
  ($68,933), which confirms that a minority of expensive properties absorb most of the total
  error. This is consistent with the right-skewed price distribution seen in the EDA.
* **R²** is scale-free and is used only to compare models against each other. On its own it says
  nothing about dollars, which is why it is not the headline.

## How I built it

### Baseline, four models at default settings

Minimal preparation: `id` and `date` dropped, `zipcode` left as `int64`, no encoding, no
engineered features, no tuning. 18 features.

| Model | R² | RMSE | MAE |
|---|---|---|---|
| **Random Forest** | **0.8837** | 128,607 | **69,815** |
| Gradient Boosting | 0.8762 | 132,716 | 78,008 |
| Decision Tree | 0.7743 | 179,155 | 98,545 |
| Linear Regression | 0.7103 | 202,979 | 125,521 |

The spread between Linear Regression and the tree ensembles is the story here. A linear model
cannot use a numeric zipcode meaningfully at all. A tree can approximate it by splitting the
numeric range into groups — clumsier than one-hot encoding, since it can only make contiguous
cuts, but enough to recover most of the location signal. Random Forest also beats the single
Decision Tree by a wide margin, which is bagging doing its job: averaging 100 trees cancels the
overfitting a single unpruned tree suffers from.

Random Forest leads on both metrics and is carried forward.

### Tuning, applied at every stage rather than only at the end

`RandomizedSearchCV`, 5-fold CV, `scoring='neg_mean_squared_error'`, `n_iter=10`, at most 3 values
per hyperparameter, `random_state=42` throughout.

```
n_estimators:      [100, 200, 300]
max_depth:         [10, 20, None]
min_samples_split: [2, 5, 10]
```

The same search space is reused at **every** iteration. This is deliberate: if only the final
model were tuned, any improvement over the baseline would be a mix of better features and more
search effort, with no way to separate them. Tuning both sides means every comparison in this
project is tuned-against-tuned, so a gain is attributable to the features.

Tuning the baseline Random Forest improved R² from 0.8837 to 0.8861 and reduced MAE from $69,815
to $69,321, a gain of +0.0024 R² and $493.

### Iteration 1, one-hot encoding and feature engineering

`zipcode` cast to string and one-hot encoded, expanding 18 features to 88. Two engineered features
added: `house_age` (derived from `yr_built`) and `is_renovated` (a yes/no flag replacing the
year-zero sentinel in `yr_renovated`).

| | R² | RMSE | MAE |
|---|---|---|---|
| Baseline (tuned) | 0.8861 | 127,262 | 69,321 |
| Iteration 1 (default) | 0.8857 | 127,517 | 69,264 |
| **Iteration 1 (tuned)** | **0.8861** | 127,305 | **68,933** |

R² is unchanged to four decimal places. The entire gain shows up in MAE, which drops by $388
against the tuned baseline.

That split between the two metrics is the most useful result in the project. R² measures the
fraction of *variance* explained, and variance is computed from squared deviations, so a house
that is $400k off contributes 100× more than one $40k off. King County's price distribution has a
long right tail, and errors on those few expensive homes dominate R² entirely. MAE weights every
dollar equally. **The encoding and engineered features helped on ordinary homes and did nothing
for the expensive tail.**

On the engineered features specifically, the small movement is expected for a tree model.
`house_age` is a monotonic transformation of `yr_built` — a split at "yr_built > 1990" is the
identical split as "house_age < 25", so the tree already had that information. The same applies to
`is_renovated`, since the tree could already split `yr_renovated` at 0. `is_renovated` is retained
regardless because it fixes a real data problem, not because it moves the metric.

### Iteration 2, feature importance and selection, tested and rejected

Feature importance on the Iteration 1 model showed **62 of 88 features carrying near-zero
importance** (individually below 0.0005). The obvious hypothesis is that most of the 69 zipcode
dummies are dead weight and the model would be simpler and no worse without them.

Selecting the top 20 features by importance and retraining:

| Feature set | R² | MAE |
|---|---|---|
| All 88 features | 0.8857 | 69,264 |
| Top 20 (default) | 0.8837 | 69,789 |
| Top 20 (tuned) | 0.8852 | 69,462 |

**Selection made things worse and was rejected.** Tuning recovered part of the loss but never
closed the gap.

The reason is `max_features`. Random Forest considers only a random subset of features at each
split, so when the strong features are not in that subset, a weak feature gets used and
contributes something real. Feature importance measures a feature's *average* contribution across
all trees; it does not measure what happens when the feature is *absent*. Sixty-two features that
each look negligible in isolation were collectively carrying information the ensemble was using.

### Everything tried, in one table

| Stage | Features | Model | R² | RMSE | MAE |
|---|---|---|---|---|---|
| Baseline | 18 raw | Random Forest | 0.8837 | 128,607 | 69,815 |
| Baseline | 18 raw | Gradient Boosting | 0.8762 | 132,716 | 78,008 |
| Baseline | 18 raw | Decision Tree | 0.7743 | 179,155 | 98,545 |
| Baseline | 18 raw | Linear Regression | 0.7103 | 202,979 | 125,521 |
| Baseline | 18 raw | Random Forest (tuned) | 0.8861 | 127,262 | 69,321 |
| Iteration 1 | 88 encoded + engineered | Random Forest | 0.8857 | 127,517 | 69,264 |
| **Iteration 1** | **88 encoded + engineered** | **Random Forest (tuned)** | **0.8861** | **127,305** | **68,933** |
| Iteration 2 | top 20 | Random Forest | 0.8837 | 128,633 | 69,789 |
| Iteration 2 | top 20 | Random Forest (tuned) | 0.8852 | 127,764 | 69,462 |

## What I actually learned

**R² was effectively saturated at the baseline.** An untuned Random Forest on 18 raw columns
reached 0.8837. Everything afterwards — 69 zipcode dummies, two engineered features, three tuning
runs across three stages — bought 0.0024 R² and about $880 of MAE, roughly a 1.3% reduction in
typical dollar error.

That pattern suggests the model is close to the **irreducible error** for this feature set. Two
identical houses on the same street sell for different prices: one seller was in a hurry, one
buyer overpaid, one sale included the furniture. None of that is in these 18 columns and no model
can recover it. The realistic path to a materially better model is **more features, not more
modelling** — school district, crime rate, days on market, renovation quality, proximity to
transit.

Stating this is more useful than manufacturing an improvement narrative. The iterations did what
they were supposed to do; the data had less headroom than expected.

## The app

**King County Valuations**, a dark navy and brass Streamlit app styled to look like a standalone
product rather than a Streamlit demo.

The sidebar takes property details grouped into Location, Dimensions, Condition and History, with
help text explaining domain conventions such as the King County construction grade scale (3–6
below average, 7 average, 11–13 luxury) and fractional bathrooms. The main panel returns:

* A **headline valuation** with a **typical range** of ±$68,933, the model's MAE. A bare point
  estimate implies a precision the model does not have; showing the range communicates the actual
  uncertainty in the units the user cares about
* **Supporting figures**: price per sqft, configuration, year built and grade
* **Market context charts**: where this valuation sits in the local price distribution for the
  selected zipcode, and living area against price with this property marked, plus the number of
  comparable sales within ±10% of the entered living area
* A **valuation log** that accumulates across the session, exportable as CSV
* **Input validation** that rejects physical impossibilities — a basement larger than the total
  living area, or a renovation year earlier than the year built. The model would happily return a
  number for either, because it extrapolates from whatever it is given, however nonsensical

### Keeping the app and the notebook in agreement

The app loads two artefacts:

* **`model.pkl`**, the tuned Random Forest, saved with `compress=3` (292 MB uncompressed, 61.8 MB
  compressed, under GitHub's 100 MB limit)
* **`model_columns.pkl`**, the exact 88 training columns *in order*. scikit-learn matches features
  by position, so a live input row with the right columns in the wrong order would predict
  silently and wrongly. The app runs `get_dummies` on the single input row — which produces only
  one zipcode column — then reindexes onto the saved list, which creates the 87 absent columns
  filled with 0, drops anything unexpected, and fixes the order

Paths are resolved with `Path(__file__).parent` rather than bare filenames, because Streamlit
Cloud runs from the repository root rather than the script's directory. Dependency versions in
`requirements.txt` are pinned to the local environment, since unpickling a scikit-learn estimator
across major versions is not safe.

## Repository structure

```
├── houseprice_prediction.ipynb   full analysis: EDA, baseline, 2 iterations, evaluation
├── streamlitapp.py               the King County Valuations app
├── model.pkl                     the deployed tuned Random Forest (compressed)
├── model_columns.pkl             the 88 training columns, in order
├── zipcode_lookup.csv            per-zipcode medians for features the user cannot know
├── kc_house_data.csv             King County House Sales dataset
└── requirements.txt              pinned to the training environment
```

## Notebook structure

1. **Business Understanding** — the goal, who it serves, why it matters, the problem type
2. **Data Understanding** — load, summary statistics, target and feature distributions, outliers, correlation, pairplot
3. **Data Preparation** — duplicate removal, impossible values (a 33-bedroom listing, zero-bathroom rows)
4. **Modelling** — baseline (4 algorithms, metric justification, tuning), Iteration 1 (encoding, feature engineering, tuning), Iteration 2 (feature importance, selection, tuning), full comparison
5. **Predicting with new data** — a single unseen property through the full pipeline
6. **Summary** — stage-by-stage table, final model selection, business interpretation

## Limitations

* **The app substitutes zipcode medians for `lat`, `long`, `sqft_living15` and `sqft_lot15`.** The
  model was trained on exact coordinates and actual neighbour statistics, but a user valuing a
  hypothetical house does not know its latitude to four decimal places. Every house in a given
  zipcode therefore receives identical location features at inference time, collapsing the model's
  location resolution from street-level to zipcode-level. Real-world accuracy is somewhat worse
  than the $68,933 test MAE suggests
* **The ±MAE range is not a true prediction interval.** MAE is an average error, not a
  distribution. A proper interval would come from the spread of individual tree predictions within
  the forest
* **The data covers May 2014 to May 2015 only.** No temporal validation was performed and no
  inflation adjustment applied, so the model should not be assumed to transfer to a later market
* **Errors are concentrated in the expensive tail.** RMSE at 1.8× MAE confirms this. The model is
  reliable on typical homes and least reliable exactly where the stakes are highest
* **The model is 61.8 MB compressed**, which makes cold starts on Streamlit's free tier slow. A
  smaller configuration (fewer trees, a minimum leaf size) would trade a small amount of accuracy
  for much faster loading, and would be the sensible choice for real deployment
