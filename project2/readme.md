# Unsupervised Learning: Human Activity Recognition (HAR)

## 📌 Project Overview
This project applies **Unsupervised Machine Learning** techniques (Dimensionality Reduction and Clustering) to the **Human Activity Recognition Using Smartphones** dataset.

The goal is to determine if sensor data (accelerometer & gyroscope) contains enough distinct patterns to group human activities (e.g., Walking vs. Standing) **without** using labeled training data.

---

## 📂 The Dataset
* **Source:** [UCI Machine Learning Repository - HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
* **Content:** Recordings of 30 subjects performing activities of daily living (ADL) while carrying a waist-mounted smartphone.
* **Dimensions:** 10,299 Samples × 561 Features (Time & Frequency domain variables).
* **Classes (Ground Truth):** 1. Walking, 2. Walking Upstairs, 3. Walking Downstairs, 4. Sitting, 5. Standing, 6. Laying.

---

## ⚙️ Methodology & Workflow

### **Phase 1: Data Preprocessing**
* **Data Loading:** Merged the `Train` (70%) and `Test` (30%) sets to create a unified dataset of 10,299 samples for clustering.
* **Cleaning:** Handled a known issue in the raw dataset where 42 feature names were duplicates (e.g., `bandsEnergy()`). A script was implemented to rename these duplicates to ensure uniqueness.
* **Scaling:** Applied `StandardScaler` to normalize all 561 features to unit variance ($\mu=0, \sigma=1$). This was critical to prevent high-magnitude features from dominating the distance calculations in PCA and K-Means.

### **Phase 2: Dimensionality Reduction (PCA)**
* **Algorithm:** Principal Component Analysis (PCA).
* **2D Projection:** Reduced data to 2 dimensions. **Explained Variance: ~56%**.
* **3D Projection:** Reduced data to 3 dimensions. **Explained Variance: ~60%**.
* **Observation:** While distinct clusters began to form, significant overlap remained between static activities (Sitting/Standing) in the linear projection.

### **Phase 3: Clustering (K-Means)**
* **Algorithm:** K-Means Clustering ($k=6$).
* **Evaluation:** Used a Cross-Tabulation (Confusion Matrix) to compare the unsupervised cluster labels against the hidden ground truth.
* **Key Insight:**
    * **Success:** Perfect separation between **Dynamic** (Walking) and **Static** (Sitting/Laying) activities.
    * **Success:** "Laying" was identified as a distinct, isolated cluster.
    * **Limitation:** The model struggled to distinguish "Sitting" from "Standing" due to the similarity in sensor orientation.

### **Phase 4: Advanced Visualization (t-SNE)**
* **Algorithm:** t-Distributed Stochastic Neighbor Embedding (t-SNE).
* **Goal:** To capture non-linear relationships that PCA missed.
* **Result:** t-SNE successfully visually separated the "Sitting" and "Standing" clusters, demonstrating that the data is separable with manifold learning techniques, even where linear methods (PCA) failed.

---

## 📊 Key Results
| Activity Type | Unsupervised Detection Capability |
| :--- | :--- |
| **Laying** | ✅ **Excellent** (Perfectly isolated cluster) |
| **Walking (All types)** | ✅ **Good** (Separated from static, but variations like 'Upstairs' overlap) |
| **Sitting vs. Standing** | ⚠️ **Challenging** (Mixed in K-Means, but separable with t-SNE) |

## 🛠 Tools Used
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn