# Credit Card Fraud Detection

Classifying fraudulent transactions on the ULB dataset. The interesting problem here isn't accuracy: it's what to do when 99.8% of your data is the negative class.

## Problem

The dataset has 284,807 transactions; 492 are fraud. A model that predicts "not fraud" on everything gets 99.8% accuracy and catches zero fraud cases. So accuracy is useless here. The real question is: at what threshold do you maximize recall without drowning the ops team in false positives?

## Dataset

[Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) via Kaggle. Not included, download `creditcard.csv` and place it in `data/`.

## Approach

- Stratified train/val/test split to preserve class ratios
- Trained Logistic Regression, Decision Tree, and Random Forest
- Evaluated on ROC-AUC, F1 (fraud class), recall, and balanced accuracy
- Explored threshold tuning to shift the precision/recall tradeoff

## Key findings

- Random Forest outperforms the others at default threshold
- Lowering the decision threshold significantly boosts recall at the cost of precision. The right choice depends on the business context
- Ensemble methods handle imbalance better, but threshold tuning matters more than model choice

Full analysis in `credit-card-fraud-ml.ipynb`. Rendered HTML version also included.

## Run it

```bash
pip install numpy pandas scikit-learn matplotlib
jupyter notebook credit-card-fraud-ml.ipynb
```

## Stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Jupyter

---

Oussama Elmir · [github.com/osamaPY](https://github.com/osamaPY)
