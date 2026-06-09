import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def load_dataset(source):
    if isinstance(source, str):
        return pd.read_csv(source)
    return pd.read_csv(source)


def prepare_data(df: pd.DataFrame):
    df = df.copy()
    if 'Amount' not in df.columns or 'Time' not in df.columns or 'Class' not in df.columns:
        raise ValueError('Dataset must contain Amount, Time, and Class columns.')

    scaler = StandardScaler()
    df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_Scaled'] = scaler.fit_transform(df[['Time']])
    df = df.drop(['Amount', 'Time'], axis=1)
    return df


def balance_data(df: pd.DataFrame, target='Class', sample_size=10000):
    majority = df[df[target] == 0]
    minority = df[df[target] == 1]
    if len(minority) == 0 or len(majority) == 0:
        raise ValueError('Dataset must contain both fraud and legitimate transactions.')

    minority_upsampled = resample(
        minority,
        replace=True,
        n_samples=sample_size,
        random_state=42,
    )
    majority_downsampled = resample(
        majority,
        replace=False,
        n_samples=sample_size,
        random_state=42,
    )
    balanced = pd.concat([majority_downsampled, minority_upsampled])
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced


def split_data(df: pd.DataFrame, target='Class'):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def build_rf():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )


def train_models(X_train, y_train, X_test, y_test):
    rf_model = build_rf()
    rf_model.fit(X_train, y_train)

    gbc_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
    )
    gbc_model.fit(X_train, y_train)

    return {
        'rf': rf_model,
        'gbc': gbc_model,
    }


def evaluate_models(models, X_test, y_test):
    rf_proba = models['rf'].predict_proba(X_test)[:, 1]
    rf_pred = models['rf'].predict(X_test)
    gbc_proba = models['gbc'].predict_proba(X_test)[:, 1]
    gbc_pred = models['gbc'].predict(X_test)
    agent_proba = 0.4 * rf_proba + 0.6 * gbc_proba
    agent_pred = (agent_proba >= 0.5).astype(int)

    def make_report(name, true, pred, proba):
        return {
            'name': name,
            'accuracy': accuracy_score(true, pred),
            'precision': precision_score(true, pred, zero_division=0),
            'recall': recall_score(true, pred, zero_division=0),
            'f1_score': f1_score(true, pred, zero_division=0),
            'roc_auc': roc_auc_score(true, proba),
            'classification_report': classification_report(true, pred, target_names=['Legitimate', 'Fraud'], zero_division=0),
        }

    return {
        'rf': make_report('Random Forest', y_test, rf_pred, rf_proba),
        'gbc': make_report('Gradient Boosting', y_test, gbc_pred, gbc_proba),
        'agent': make_report('Fraud Agent', y_test, agent_pred, agent_proba),
    }


def summarize_dataset(df: pd.DataFrame):
    return {
        'rows': len(df),
        'columns': df.shape[1],
        'fraud_count': int((df['Class'] == 1).sum()),
        'legit_count': int((df['Class'] == 0).sum()),
        'fraud_percentage': float((df['Class'] == 1).sum() / len(df) * 100),
    }


def predict_agent(models, X):
    rf_proba = models['rf'].predict_proba(X)[:, 1]
    gbc_proba = models['gbc'].predict_proba(X)[:, 1]
    agent_proba = 0.4 * rf_proba + 0.6 * gbc_proba
    predictions = (agent_proba >= 0.5).astype(int)
    return predictions, agent_proba
