"""
Phase 1 — TP LSTM : Airline Passengers, data et sliding window.

Objectif : charger la série temporelle, l'analyser, la normaliser, et la
transformer en paires (X, y) via fenêtre glissante.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


AIRLINE_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "airline-passengers.csv"
)
WINDOW_SIZE = 12


def load_airline_data(url=AIRLINE_URL):
    """Charge la série Airline Passengers (colonne 'Passengers')."""
    df = pd.read_csv(url)
    series = df["Passengers"].values.astype("float32").reshape(-1, 1)
    print(f"Shape brute : {series.shape}")
    print(f"Min : {series.min():.1f}, Max : {series.max():.1f}, "
          f"Moyenne : {series.mean():.1f}")
    return series


def normalize_series(series):
    """Normalise la série entre 0 et 1 avec MinMaxScaler.

    Le LSTM est sensible aux grandes valeurs : les sigmoid et tanh internes
    saturent dès qu'on dépasse ~5. Toujours normaliser une série temporelle
    avant de la donner à un RNN.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    series_scaled = scaler.fit_transform(series)
    return series_scaled, scaler


def temporal_split(series, train_ratio=0.67):
    """Split chronologique : jamais de shuffle sur une série temporelle.

    Le train est la première portion (passé), le test est la fin (futur).
    Mélanger reviendrait à entraîner sur le futur et prédire le passé
    (data leakage).
    """
    n_train = int(len(series) * train_ratio)
    train = series[:n_train]
    test = series[n_train:]
    print(f"Train shape : {train.shape}, Test shape : {test.shape}")
    return train, test


def create_dataset(dataset, window_size=WINDOW_SIZE):
    """
    Transforme une série 1D en paires (X, y) via sliding window.

    Pour chaque position i :
        X[i] = dataset[i : i+window_size]
        y[i] = dataset[i + window_size]

    Retourne :
        X : shape (N - window_size, window_size, 1)  -- dim 1 = features
        y : shape (N - window_size,)
    """
    # Garde-fou adversarial : NaN dans la série
    if np.isnan(dataset).any():
        raise ValueError(
            "La série contient des NaN. Imputer ou supprimer avant d'appeler "
            "create_dataset."
        )

    X, y = [], []
    for i in range(len(dataset) - window_size):
        X.append(dataset[i:i + window_size, 0])
        y.append(dataset[i + window_size, 0])

    X = np.array(X).reshape(-1, window_size, 1).astype("float32")
    y = np.array(y).astype("float32")
    return X, y


def plot_series(train, test, title="Airline Passengers — split train/test"):
    """Visualise les courbes train et test : pas de chevauchement temporel."""
    plt.figure(figsize=(12, 4))
    plt.plot(np.arange(len(train)), train, label="Train")
    plt.plot(
        np.arange(len(train), len(train) + len(test)), test,
        label="Test", color="orange",
    )
    plt.title(title)
    plt.xlabel("Mois")
    plt.ylabel("Passagers (normalisés)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Pipeline phase 1
    series = load_airline_data()
    series_scaled, scaler = normalize_series(series)
    train, test = temporal_split(series_scaled, train_ratio=0.67)
    plot_series(train, test)

    X_train, y_train = create_dataset(train, WINDOW_SIZE)
    X_test, y_test = create_dataset(test, WINDOW_SIZE)

    print(f"\nX_train : {X_train.shape}")  # attendu : (84, 12, 1)
    print(f"y_train : {y_train.shape}")    # attendu : (84,)
    print(f"X_test  : {X_test.shape}")     # attendu : (36, 12, 1)
    print(f"y_test  : {y_test.shape}")

    # ===== Quality gates =====

    # Edge case : window_size=1
    print("\n--- Edge case : window_size=1 ---")
    X1, y1 = create_dataset(train, window_size=1)
    print(f"X shape : {X1.shape}, y shape : {y1.shape}")
    print(
        "Avec 1 step de contexte, le modèle ne voit que la valeur précédente.\n"
        "Il perd la saisonnalité (cycle annuel sur 12 mois) → RMSE plus élevé."
    )

    # Adversarial : NaN dans la série
    print("\n--- Adversarial : NaN dans la série ---")
    series_with_nan = series_scaled.copy()
    series_with_nan[5, 0] = np.nan
    try:
        create_dataset(series_with_nan)
    except ValueError as e:
        print(f"Erreur capturée correctement : {e}")
