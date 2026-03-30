# 📊 SORA Analytics — Liquidity, FX, and Market Structure

A data-driven exploration of the **Singapore Overnight Rate Average (SORA)**, focusing on liquidity dynamics, calendar effects, and macro-financial relationships.

---

## 🧠 Overview

This project analyzes SORA using historical data, with an emphasis on:

* Understanding **liquidity conditions**
* Identifying **statistical patterns**
* Exploring **calendar and structural effects**
* Building toward **predictive/policy insights**

---

## 📦 Data Sources

* SORA historical data (MAS): https://eservices.mas.gov.sg/statistics/dir/DomesticInterestRates.aspx
* FX rates (MAS): https://eservices.mas.gov.sg/statistics/msb/exchangerates.aspx
* US Federal Reserve rates: https://www.macrotrends.net/datasets/2015/fed-funds-rate-historical-chart

---

# 🧪 Phase 1 — Data Exploration & Analysis

## 🔍 Objective

Understanding what lies in plain sight. Before deeper analysis & predictive models. 

---

## 📊 1. Distribution of SORA Changes

**Method**

I analyze the distribution of daily changes in SORA with a simple .describe()

**Result**

```
Mean: ~0  
Std: 0.151  
Min/Max: -1.26 / +1.48  
Skew: +0.165
```

**Interpretation**

* Changes are centered around zero (low mean/std)
* Presence of **fat tails** (large but rare movements)
* Slight bias toward **upward spikes** (slight positive skew)

**Takeaway**

> SORA is typically stable but exhibits occasional large upward spikes.

---

## 🔁 2. Autocorrelation & Mean Reversion

**Method**

I compute lag-1 autocorrelation of SORA changes.

**Result**

```
Autocorrelation: -0.063
```

**Interpretation**

* Weak negative autocorrelation
* Indicates **mild mean reversion**

**Takeaway**

> Short-term SORA movements tend to partially reverse. If it goes up today it's slightly more likely to go back down tomorrow. 

---

## 📊 3. Liquidity Indicators (Range & Volume)

**Method**

I examine relationships between SORA and:

* Derived Intraday range (High - Low)
* Transaction volume

**Result**

```
Correlation (Range vs SORA): 0.755  
Correlation (Volume vs SORA): 0.25
```

**Interpretation**

* Strong relationship between **rate dispersion and SORA**
* Volume has weaker but noticeable influence

**Takeaway**

> Liquidity stress (captured by intraday range) is a **primary driver** of SORA movements.

---

## 📅 4. Calendar Effects

### Weekday Analysis

**Method**

* Group SORA changes by weekday
* Perform statistical tests (t-test, ANOVA)

**Result**

```
Day of Week
Mon   -0.025125
Tue   -0.021063
Wed   -0.015225
Thu    0.001196
Fri    0.062349
```

```
ANOVA: F = 38.92, p < 0.001 (6.92e-32)
T-test (Friday vs others): p < 0.001 (2.65e-32)
```

**Interpretation**

* Significant differences across weekdays
* Friday shows higher average increases

---

### ⚡ Spike Analysis (Extreme Movements)

**Method**

I identify extreme SORA changes (defined here as >2 standard deviations) and analyze their distribution across weekdays.

**Result**

```
Chi-square test: χ² = 10.30, p = 0.036  

Positive spikes:
- Friday    : 46.7%
- Thursday  : 23.9%
- Monday    : 10.8%
- Tuesday   : 10.8%
- Wednesday : 7.6%

Negative spikes:
- Monday    : 24.4%
- Tuesday   : 23.4%
- Thursday  : 23.4%
- Wednesday : 20.4%
- Friday    : 8%
```

**Interpretation**

* Upward spikes incredibly concentrated at **end of week** (70.6%)
* Downward adjustments occur **at start of week** (47.8%)

**Takeaway**

> SORA exhibits a **weekly liquidity cycle**, where funding conditions tighten before the weekend and normalize afterward. Before the weekend liquidity tightens to cover 3 days of exposure (Sat, Sun, Mon) and loosens on Monday & Tuesday. 

---

## ⚡ 5. Spike Behavior

**Method**

I identify extreme movements defined here as a 2σ threshold.

**Result**

```text
Number of spikes: 190 out of ~3,300 observations (roughly 5.7%)
```

**Interpretation**

* Relatively frequent extreme events relative to normal distribution. **One can expect 1-2 spikes per month.**

**Takeaway**

> SORA dynamics are characterized by **fat tails and episodic stress events**.

---

## 🧪 6. Stationarity

**Method**

Augmented Dickey-Fuller (adfuller) test on SORA changes.

**Result**

```text
ADF Statistic: -16.08  
p-value: 5.38e-29
```

**Interpretation**

* Strong rejection of non-stationarity

**Takeaway**

> SORA changes are **stationary**, making them suitable for time-series modeling.

---

# 🧠 Key Findings

* SORA is **stable most of the time**, with occasional large shocks
* Liquidity conditions (range) strongly influence rate movements
* A **weekly liquidity cycle** exists around weekends
* SORA exhibits **mean reversion and volatility clustering**
* Extreme events are more frequent than expected (fat tails)


# ⚠️ Disclaimer

This project is for educational and research purposes only.
It does not constitute financial advice.

---

# 🛠️ Tech Stack

* Python (Pandas, NumPy, SciPy)
* Statsmodels
* Scikit-learn
* Jupyter Notebook

---
