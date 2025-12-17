# Minimizing Revenue Loss: Strategic Prediction of Hotel Booking Cancellation Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)](https://scikit-learn.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hotelbookingdemandcapstone-hpslxcled9w5yfvcfmamjw.streamlit.app/)
[![Status](https://img.shields.io/badge/Status-Completed-green)]()

## Project Overview
Hotel rooms are **perishable inventory**. Jika kamar kosong pada malam tertentu akibat pembatalan mendadak, pendapatan dari kamar tersebut hilang selamanya (*Revenue Leakage*).

Proyek ini bertujuan membangun model **Machine Learning (Classification)** untuk memprediksi apakah seorang tamu akan membatalkan pesanan (`is_canceled = 1`). Dengan prediksi ini, manajemen hotel dapat menerapkan strategi **Aggressive Overbooking** yang terukur untuk memaksimalkan **Occupancy Rate** dan **Revenue**.

### Business Problem
- **Inventory Spoilage (Kamar Kosong):** Hotel menahan kamar untuk tamu yang ternyata batal. Kerugian: **~$200** (High Season Opportunity Cost).
- **Inventory Spill (Overbooking):** Hotel menjual kamar terlalu banyak, tamu datang semua, hotel harus memindahkan tamu (*Walk the Guest*). Biaya: **~$75** (Kompensasi & Transport).

**Strategic Decision:** Karena biaya *False Negative* (Kamar Kosong) **2.7x lebih mahal** daripada *False Positive* (Overbooking), model ini dioptimalkan untuk **Recall** (menangkap potensi batal seagresif mungkin) menggunakan **F2-Score** dan **Threshold Tuning**.

---

## Dataset & Features
Dataset terdiri dari **83.573 baris** data transaksi pemesanan hotel (CRM) dengan 11 fitur utama.

| Feature | Description |
| :--- | :--- |
| `country` | Negara asal tamu (High Cardinality). |
| `market_segment` | Channel pemesanan (Online TA, Offline TA, Direct, Corporate, dll). |
| `deposit_type` | Tipe deposit (No Deposit, Non Refund, Refundable). |
| `previous_cancellations` | Jumlah pembatalan historis oleh tamu. |
| `booking_changes` | Jumlah perubahan yang dilakukan pada booking. |
| `days_in_waiting_list` | Lama hari booking berada di waiting list. |
| `total_of_special_requests` | Jumlah permintaan khusus (bantal tambahan, view, dll). |
| `is_canceled` | **Target Variable** (1 = Cancel, 0 = Not Cancel). |

---

## Methodology

### 1. Data Preprocessing
- **Missing Values:** Mengisi `country` yang kosong dengan "Other" (menghindari bias modus).
- **Handling Duplicates:** Data duplikat dipertahankan (karena tidak ada Unique ID dan transaksi identik dianggap valid).
- **Cardinality Reduction:** Mengelompokkan negara menjadi `PRT` (Portugal), `Top_International`, dan `Other`.

### 2. Feature Engineering (Key Innovation)
Fitur baru diciptakan untuk menangkap "Niat & Risiko" tamu:
- **`commitment_score`**: Gabungan dari *Special Requests* + *Parking* + *Booking Changes*. (Skor tinggi = Niat menginap kuat).
- **`is_high_risk`**: *Flagging* tamu dengan deposit 'Non Refund' (tapi bukan Corporate) atau yang punya riwayat cancel.
- **`booking_stability`**: Mengukur kestabilan pesanan berdasarkan perubahan vs durasi waiting list.

### 3. Model Selection
Membandingkan beberapa algoritma (Logistic Regression, KNN, Decision Tree, XGBoost, dll).
- **Chosen Model:** **Bagging Classifier** (Base Estimator: Decision Tree).
- **Why?** Memberikan performa **F2-Score** tertinggi dan paling stabil (Low Variance) dibandingkan Single Decision Tree.

### 4. Threshold Optimization
Alih-alih menggunakan batas standar 0.50, kami melakukan *tuning* ambang batas probabilitas.
- **New Threshold:** **0.27**
- **Impact:** Jika model mendeteksi probabilitas batal > 27%, sistem langsung menandainya sebagai "Prediksi Batal". Ini meningkatkan sensitivitas terhadap risiko *No-Show*.

---

## Key Insights (EDA & SHAP)
Berdasarkan analisis data dan *Model Explainability* (SHAP):
1.  **Online TA Risk:** Booking via *Online Travel Agent* yang masuk *waiting list* memiliki tingkat pembatalan hampir **100%**.
2.  **Commitment Score:** Tamu yang "merepotkan" (banyak request/ubah booking) justru lebih **loyal** dan jarang membatalkan.
3.  **Deposit Paradox:** Tipe deposit *Non-Refund* pada dataset ini justru memiliki tingkat cancel tinggi (kemungkinan promo murah yang ditinggalkan tamu).

---

## Business Impact Analysis
Simulasi dilakukan pada data test (16,715 booking) dengan asumsi biaya FN=\$200 dan FP=\$75.

| Skenario | Deskripsi | Estimasi Kerugian |
| :--- | :--- | :--- |
| **Without Model** | Pasif (anggap semua tamu datang). Banyak kamar kosong. | **-$1,231,200** |
| **With Tuned Model** | Agresif (prediksi batal & jual ulang kamar). | **-$455,550** |
| **TOTAL SAVING** | **Potensi Pendapatan yang Diselamatkan** | **+$775,650 (~63%)** |

---

## Recommendations
1.  **High Season Strategy:** Terapkan *Aggressive Overbooking*. Risiko membayar kompensasi jauh lebih kecil dibanding membiarkan kamar kosong saat harga tinggi.
2.  **Soft Confirmation:** Gunakan prediksi model untuk mengirim pesan otomatis (WhatsApp/Email) kepada tamu berisiko tinggi 3-7 hari sebelum kedatangan untuk konfirmasi ulang.
3.  **Operational Policy:** Hilangkan opsi *waiting list* untuk channel **Online TA** karena konversinya sangat rendah.

---

## Tools & Libraries
*   **Language:** Python
*   **Data Manipulation:** Pandas, NumPy
*   **Visualization:** Matplotlib, Seaborn
*   **Machine Learning:** Scikit-Learn (Bagging, Decision Tree, RandomizedSearchCV), Imbalanced-Learn
*   **Interpretability:** SHAP

---
*Presented by Alifsya Salam*
