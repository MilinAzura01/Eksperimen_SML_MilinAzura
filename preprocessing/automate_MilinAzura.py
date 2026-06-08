"""
automate_MilinAzura.py
======================
Script otomatisasi preprocessing dataset Titanic.
Jalankan: python automate_MilinAzura.py [raw_filepath] [output_dir]

Contoh:
    python automate_MilinAzura.py ../titanic_raw/Titanic-Dataset.csv titanic_preprocessing
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset CSV dari filepath yang diberikan."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File tidak ditemukan: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Data loaded dari '{filepath}'")
    print(f"       Shape: {df.shape[0]} baris x {df.shape[1]} kolom")
    return df


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop kolom yang tidak relevan untuk modelling."""
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing)
    print(f"[INFO] Kolom di-drop: {existing}")
    print(f"       Shape setelah drop: {df.shape}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values pada dataset."""
    if 'Age' in df.columns and df['Age'].isnull().any():
        median_age = df['Age'].median()
        df['Age'] = df['Age'].fillna(median_age)
        print(f"[INFO] Age: diisi median = {median_age:.1f}")

    if 'Embarked' in df.columns and df['Embarked'].isnull().any():
        mode_emb = df['Embarked'].mode()[0]
        df['Embarked'] = df['Embarked'].fillna(mode_emb)
        print(f"[INFO] Embarked: diisi modus = '{mode_emb}'")

    print(f"[INFO] Total missing values setelah handling: {df.isnull().sum().sum()}")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Buat fitur baru berdasarkan fitur yang ada."""
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    print("[INFO] Feature engineering selesai: FamilySize, IsAlone ditambahkan")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Encode fitur kategorikal menggunakan LabelEncoder."""
    le = LabelEncoder()
    if 'Sex' in df.columns:
        df['Sex'] = le.fit_transform(df['Sex'])
        print("[INFO] Sex encoded: female=0, male=1")
    if 'Embarked' in df.columns:
        df['Embarked'] = le.fit_transform(df['Embarked'])
        print("[INFO] Embarked encoded: C=0, Q=1, S=2")
    return df


def scale_numerical(df: pd.DataFrame) -> tuple:
    """Scale fitur numerik menggunakan StandardScaler."""
    scaler = StandardScaler()
    num_cols = [c for c in ['Age', 'Fare', 'FamilySize'] if c in df.columns]
    df[num_cols] = scaler.fit_transform(df[num_cols])
    print(f"[INFO] StandardScaler diterapkan pada: {num_cols}")
    return df, scaler


def split_data(df: pd.DataFrame, target_col: str = 'Survived',
               test_size: float = 0.2, random_state: int = 42) -> tuple:
    """Split dataset menjadi train dan test set."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Train-test split selesai:")
    print(f"       X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def save_result(X_train, X_test, y_train, y_test,
                output_dir: str = 'titanic_preprocessing') -> None:
    """Simpan hasil preprocessing ke folder output."""
    os.makedirs(output_dir, exist_ok=True)
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df  = pd.concat([X_test,  y_test],  axis=1)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'),  index=False)
    print(f"[INFO] File tersimpan di folder: {output_dir}")
    print(f"       train.csv: {train_df.shape}")
    print(f"       test.csv : {test_df.shape}")


def preprocess_pipeline(raw_filepath: str,
                        output_dir: str = 'titanic_preprocessing') -> tuple:
    """Pipeline lengkap preprocessing Titanic dataset."""
    print("=" * 50)
    print("  PREPROCESSING PIPELINE - Titanic Dataset")
    print("=" * 50)

    df = load_data(raw_filepath)
    df = drop_irrelevant_columns(df)
    df = handle_missing_values(df)
    df = feature_engineering(df)
    df = encode_categorical(df)
    df, _ = scale_numerical(df)
    X_train, X_test, y_train, y_test = split_data(df)
    save_result(X_train, X_test, y_train, y_test, output_dir)

    print("=" * 50)
    print("  PIPELINE SELESAI!")
    print("=" * 50)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    raw_path = sys.argv[1] if len(sys.argv) > 1 else "../titanic_raw/Titanic-Dataset.csv"
    out_dir  = sys.argv[2] if len(sys.argv) > 2 else "titanic_preprocessing"

    X_train, X_test, y_train, y_test = preprocess_pipeline(raw_path, out_dir)
