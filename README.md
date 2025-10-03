# Banking-Financial-Data-Analytics-Project

Perfect 🙌 You already have a **banking financial analytics project with PySpark + ML**.
Now, I’ll help you create a **GitHub-style documentation (README.md)** that explains your project clearly for recruiters or collaborators.

Here’s a polished version:

---

# 📊 Banking Financial Data Analytics Project

## 🔹 Overview

This project simulates **bank financial operations** (customers, loans, and transactions) and applies **data analytics + machine learning** to uncover insights.
It demonstrates the **end-to-end workflow of a Data Analyst / Data Scientist** in a banking domain:

* Customer segmentation
* Loan & transaction summarization
* Data visualization
* Loan default prediction (ML model)

Built with **PySpark, SQL, Python (Scikit-learn), and visualization tools (Matplotlib, Seaborn).**

---

## 📂 Project Structure

```
Bank-Financial-Analytics/
│── data/
│   ├── customers.csv
│   ├── loans.csv
│   ├── transactions.csv
│
│── notebooks/
│   ├── Bank_Analytics_Project.ipynb
│
│── src/
│   ├── data_preprocessing.py
│   ├── visualization.py
│   ├── modeling.py
│
│── README.md
│── requirements.txt
```

---

## 🛠️ Tech Stack

* **Languages:** Python (Pandas, NumPy, Matplotlib, Scikit-Learn), SQL
* **Big Data & Processing:** Apache Spark, PySpark, SparkSQL
* **Visualization:** Matplotlib, Seaborn, Power BI / Tableau (optional)
* **ML Models:** Random Forest Classifier
* **Tools:** Jupyter Notebook, GitHub, Azure Databricks

---

## 📊 Data Description

### 1. **Customers**

| CustomerID | Name   | Age | Gender | Occupation | Income | MaritalStatus |
| ---------- | ------ | --- | ------ | ---------- | ------ | ------------- |
| C001       | John D | 34  | Male   | Engineer   | 80000  | Married       |

### 2. **Loans**

| LoanID | CustomerID | LoanAmount | InterestRate | TenureMonths | LoanStatus |
| ------ | ---------- | ---------- | ------------ | ------------ | ---------- |
| L001   | C001       | 200000     | 12.5         | 36           | Active     |

### 3. **Transactions**

| TransactionID | CustomerID | Date       | Amount | Type  |
| ------------- | ---------- | ---------- | ------ | ----- |
| T001          | C001       | 2025-01-10 | 1500   | Debit |

---

## 🚀 Workflow

### 1️⃣ Data Preprocessing

* Loaded **CSV datasets** into **PySpark DataFrames**.
* Cleaned missing values, casted data types.

```python
customers = spark.read.csv("customers.csv", header=True, inferSchema=True)
loans = loans.withColumn("LoanAmount", col("LoanAmount").cast("float"))
```

---

### 2️⃣ Data Aggregation & Summarization

Created **loan and transaction summaries** per customer:

```python
loan_summary = loans.groupBy("CustomerID") \
    .agg(
        _sum("LoanAmount").alias("TotalLoan"),
        avg("InterestRate").alias("AvgInterest"),
        count("LoanID").alias("LoanCount")
    )
```

---

### 3️⃣ Visualization & Insights

* Loan distribution:
  ![Loan Distribution](images/loan_dist.png)

* Income vs Loan Status:
  ![Income vs Loan](images/income_loan.png)

---

### 4️⃣ Machine Learning Model

**Objective:** Predict whether a loan will default.

* **Features:** Income, Age, TotalLoan, LoanCount
* **Target:** LoanStatus (Default / Non-default)
* **Model:** Random Forest Classifier

```python
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
```

✅ Output:

* **Accuracy**: ~80–85%
* **Classification Report & Confusion Matrix** included.

---

## 📈 Key Insights

1. Customers with **lower income** have a higher chance of defaulting.
2. Loan amount distribution is **skewed towards mid-sized loans (1L–5L)**.
3. Transactions reflect spending habits that correlate with repayment ability.

---

## 📌 Future Improvements

* Add **customer segmentation** (K-Means clustering).
* Create **Power BI dashboard** for executives.
* Deploy ML model as a **REST API in Azure Databricks**.

---

## 🧑‍💻 Author

**Sudipto Bhattacharya**
📌 Data Analyst | Banking & Finance | Machine Learning | Azure | Databricks

---
