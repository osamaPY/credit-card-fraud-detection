# Credit Card Fraud Detection

This repository contains a machine learning project focused on detecting fraudulent credit card transactions.

The goal is to build and evaluate classical ML models on a highly imbalanced dataset, and to understand the trade-offs involved when detecting rare but important events such as fraud.

---

## Dataset

- **Credit Card Fraud Detection Dataset (ULB)**
- Source: Kaggle  
  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset is **not included** in this repository because it is large.

**To run the project:**
1. Download `creditcard.csv` from the link above  
2. Create a `data/` folder  
3. Place the file inside `data/creditcard.csv`

---

## Project Structure

```
credit-card-fraud-detection/
│
├── credit-card-fraud-ml.ipynb
├── credit-card-fraud.html
├── README.md
├── .gitignore
│
└── data/
    └── README.md
```

---

## Approach

The project follows a standard supervised learning pipeline:
- Data loading and preprocessing
- Train / validation / test split with stratification
- Training multiple models
- Evaluating performance using metrics suited for class imbalance

### Models used
- Logistic Regression  
- Decision Tree  
- Random Forest  

### Evaluation metrics
Because fraud cases are rare, accuracy alone is misleading.  
The main metrics used are:
- ROC-AUC
- Recall (fraud class)
- F1-score
- Balanced accuracy

---

## Results

The experiments show that:
- Class imbalance strongly affects model performance
- Threshold choice has a large impact on recall
- Ensemble models generally perform better than simpler ones

Detailed results and plots are available in the notebook and in the HTML report.

---

## How to run

```bash
pip install numpy pandas scikit-learn matplotlib
jupyter notebook credit-card-fraud-ml.ipynb
```

---

## Tools

- Python
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Jupyter Notebook

---

## Author

Oussama Elmir

This project was developed for educational purposes.
