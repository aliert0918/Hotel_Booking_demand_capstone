# 🏨 Minimizing Revenue Loss: Hotel Booking Cancellation Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)](https://scikit-learn.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hotelbookingdemandcapstone-hpslxcled9w5yfvcfmamjw.streamlit.app/)
[![Status](https://img.shields.io/badge/Status-Completed-green)]()

## 📌 Project Overview
High cancellation rates significantly impact hotel revenue, leading to **Revenue Leakage** (empty rooms) or **Opportunity Loss** (turning away potential guests). This project utilizes Machine Learning to predict booking cancellations (`is_canceled`), enabling the hotel to implement a dynamic overbooking strategy.

**Key Achievement:** The optimized model successfully reduced potential revenue loss by approximately **63%**, translating to a saving of **~$775,650**.

---

## 💼 Business Understanding

### The Problem
*   **Inventory Spoilage:** Rooms remain empty due to last-minute cancellations (Revenue Leakage).
*   **Inventory Spill:** Hotel plays it too safe and refuses bookings, missing out on potential revenue (Opportunity Loss).

### Goal & Metric
*   **Objective:** Minimize **False Negatives** (Predicting guest will come, but they cancel).
*   **Primary Metric:** **F2-Score**. We prioritize **Recall** because the cost of an empty room (~$200) is significantly higher than the cost of overbooking (~$75).

### Stakeholders
*   **Revenue Manager:** Strategic inventory planning.
*   **Front Office:** Operational execution (guest handling).
*   **Reservation Team:** Booking validation.

---

## 📊 Data & Preprocessing

The dataset consists of **83,573 rows** and **11 columns**, covering booking details, customer demographics, and reservation status.

### Pipeline Steps:
1.  **Data Cleaning:**
    *   Handled missing values in `country` and `market_segment`.
    *   Kept duplicates (assumed as valid separate transactions/group bookings).
2.  **Feature Engineering:**
    *   Created `commitment_score`: A composite score of special requests & parking needs.
    *   Created `is_repeat_canceler` & `is_high_risk`: To flag risky behaviors.
3.  **Transformation:**
    *   **Numerical:** RobustScaler (to handle outliers in `lead_time`).
    *   **Categorical:** One-Hot Encoding.

### Key EDA Insight
> **"Online TA has the highest cancellation rate."**
> Guests from Online Travel Agents demand instant certainty and are prone to cancel, whereas Direct bookings show high loyalty (lowest cancellation rate: 12.5%).

---

## 🤖 Modeling Strategy

We employed **Ensemble Learning** to handle complex, non-linear patterns in the data.

### Champion Model: Bagging Classifier
*   **Why?** To reduce variance and improve stability compared to a single Decision Tree.
*   **Configuration:**
    *   Base Estimator: Decision Tree with `class_weight='balanced'`.
    *   Optimization: **RandomizedSearchCV** to find the best hyperparameters.

### Strategic Move: Threshold Optimization
Instead of the default 0.5 threshold, we tuned the decision boundary to **0.27**.
*   **Impact:** The model became "aggressive" in detecting cancellations.
*   **Result:** Significantly reduced False Negatives (Empty Rooms).

---

## 💰 Business Impact Evaluation

We simulated the financial impact on a test set using the following cost matrix:
*   **Cost of False Negative (Empty Room):** $200
*   **Cost of False Positive (Overbooking/Walk Cost):** $75

| Scenario | False Negative (Empty Rooms) | False Positive (Overbooked) | Total Estimated Loss |
| :--- | :---: | :---: | :---: |
| **Without Model** | 6,156 | 0 | **$1,231,200** |
| **With Tuned Model** | 339 | 5,170 | **$455,550** |
| **TOTAL SAVINGS** | - | - | **$775,650 (~63%)** |

> **Conclusion:** Implementing the model saves the hotel over **$775k** compared to having no prediction strategy.

---

## 💡 Recommendations

Based on the analysis, we recommend the following strategies:

1.  **Aggressive Overbooking (High Season):**
    *   When demand > $150/night, use the model to resell rooms predicted to cancel. The risk of overbooking is far cheaper than the loss of an empty room.
2.  **CRM Integration ("Soft Confirmation"):**
    *   Trigger automated WhatsApp/Email confirmations for bookings flagged as "High Risk" 3-7 days prior to arrival.
3.  **Conservative Strategy (Low Season):**
    *   When rates are low (<$80), increase the prediction threshold to avoid unnecessary walk costs.
4.  **Policy Adjustment:**
    *   Eliminate "Waiting List" option for Online TA channels as they have a ~100% churn rate on waitlists.

---

## 🛠 Tools & Libraries
*   **Language:** Python
*   **Data Manipulation:** Pandas, NumPy
*   **Visualization:** Matplotlib, Seaborn
*   **Machine Learning:** Scikit-Learn (Bagging, Decision Tree, GridSearch), Imbalanced-Learn
*   **Interpretability:** SHAP

---
*Presented by Alifsya Salam*
