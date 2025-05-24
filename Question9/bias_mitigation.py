# bias_mitigation.py
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    # Step 1: Load dataset
    print("Loading dataset...")
    adult = fetch_openml(name='adult', version=2, as_frame=True)
    df = adult.frame

    # Step 2: Preprocess
    print("Preprocessing data...")
    df.replace('?', pd.NA, inplace=True)
    df.dropna(inplace=True)

    y = (df['class'] == '>50K').astype(int)
    X = df.drop(columns=['class'])

    # Keep 'sex' for bias check
    sex = X['sex']
    X = pd.get_dummies(X.drop(columns=['sex']), drop_first=True)

    # Step 3: Train baseline model
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test, sex_train, sex_test = train_test_split(
        X, y, sex, test_size=0.2, random_state=42, stratify=y
    )

    print("Training baseline Logistic Regression model...")
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Overall Accuracy (Baseline): {accuracy_score(y_test, y_pred):.4f}")

    # Step 4: Bias check
    acc_male = accuracy_score(y_test[sex_test == 'Male'], y_pred[sex_test == 'Male'])
    acc_female = accuracy_score(y_test[sex_test == 'Female'], y_pred[sex_test == 'Female'])
    print(f"Accuracy Male (Baseline): {acc_male:.4f}")
    print(f"Accuracy Female (Baseline): {acc_female:.4f}")

    # Step 5: Bias mitigation by reweighting
    print("Applying reweighting for bias mitigation...")
    # Calculate weights inversely proportional to group frequency in training set
    group_counts = sex_train.value_counts()
    total = len(sex_train)
    weights = sex_train.apply(lambda s: total / (2 * group_counts[s]))

    # Retrain with sample weights
    model_reweighted = LogisticRegression(max_iter=200)
    model_reweighted.fit(X_train, y_train, sample_weight=weights)

    # Step 6: Evaluate reweighted model
    y_pred_rw = model_reweighted.predict(X_test)
    print(f"Overall Accuracy (Reweighted): {accuracy_score(y_test, y_pred_rw):.4f}")

    acc_male_rw = accuracy_score(y_test[sex_test == 'Male'], y_pred_rw[sex_test == 'Male'])
    acc_female_rw = accuracy_score(y_test[sex_test == 'Female'], y_pred_rw[sex_test == 'Female'])
    print(f"Accuracy Male (Reweighted): {acc_male_rw:.4f}")
    print(f"Accuracy Female (Reweighted): {acc_female_rw:.4f}")

if __name__ == "__main__":
    main()
