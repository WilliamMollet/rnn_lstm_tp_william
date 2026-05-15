"""
Phase 2 — TP LSTM : Airline Passengers, construction, entraînement, prédiction.

Objectif : LSTM en régression avec Dropout, EarlyStopping, dénormalisation
des prédictions et calcul du RMSE en unité passagers.
"""
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers
from sklearn.metrics import mean_squared_error

from phase1_sliding_window import (
    load_airline_data, normalize_series, temporal_split,
    create_dataset, WINDOW_SIZE,
)


def build_lstm_model(window_size=WINDOW_SIZE, units=64, dropout=0.2):
    """LSTM → Dropout → Dense(1). Pas d'activation finale : régression."""
    model = keras.Sequential([
        layers.LSTM(units, input_shape=(window_size, 1)),
        layers.Dropout(dropout),
        layers.Dense(1),  # 1 neurone, pas d'activation → sortie continue
    ])
    model.compile(
        optimizer="adam",      # adaptatif, converge bien sur ce type de série
        loss="mse",            # régression → MSE
        metrics=["mae"],       # MAE garde l'unité passagers (lisible)
    )
    return model


def train_model(model, X_train, y_train, epochs=100, batch_size=8):
    """EarlyStopping sur val_loss, patience=10."""
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1,
    )
    return history


def evaluate_model(model, X_train, y_train, X_test, y_test, scaler):
    """Prédit, dénormalise via scaler.inverse_transform, calcule RMSE."""
    train_pred = model.predict(X_train, verbose=0)
    test_pred = model.predict(X_test, verbose=0)

    # Repasser en unité "nombre de passagers"
    train_pred = scaler.inverse_transform(train_pred)
    test_pred = scaler.inverse_transform(test_pred)
    y_train_orig = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse_train = float(np.sqrt(mean_squared_error(y_train_orig, train_pred)))
    rmse_test = float(np.sqrt(mean_squared_error(y_test_orig, test_pred)))

    print(f"\nRMSE train : {rmse_train:.2f} passagers")
    print(f"RMSE test  : {rmse_test:.2f} passagers")
    print("(Objectif : RMSE test < 50, bonne perf < 70)")

    return train_pred, test_pred, rmse_train, rmse_test


def plot_loss(history):
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"], label="loss (train)")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Évolution de la loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_predictions(series_scaled, train_pred, test_pred, window_size, scaler):
    """
    Trace la série complète + les prédictions train et test alignées sur
    leurs indices temporels d'origine.

    - train_pred prédit train[window_size : window_size + len(train_pred)]
    - test_pred prédit test[window_size : window_size + len(test_pred)]
      ce qui correspond à series[len(train) + window_size : ...]
    """
    series_unscaled = scaler.inverse_transform(series_scaled)
    n = len(series_unscaled)

    plt.figure(figsize=(12, 5))
    plt.plot(series_unscaled, label="Données réelles", linewidth=2)

    train_start = window_size
    train_end = train_start + len(train_pred)
    train_plot = np.full(n, np.nan)
    train_plot[train_start:train_end] = train_pred.flatten()

    # train_end correspond à la fin du train ; on saute encore window_size
    # avant que la première test_pred soit produite.
    test_start = train_end + window_size
    test_end = test_start + len(test_pred)
    test_plot = np.full(n, np.nan)
    test_plot[test_start:test_end] = test_pred.flatten()

    plt.plot(train_plot, label="Prédiction train", linestyle="--")
    plt.plot(test_plot, label="Prédiction test", linestyle="--")
    plt.title("Prédictions LSTM vs ground truth — Airline Passengers")
    plt.xlabel("Mois")
    plt.ylabel("Passagers")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Recharger les données (script standalone)
    series = load_airline_data()
    series_scaled, scaler = normalize_series(series)
    train, test = temporal_split(series_scaled, train_ratio=0.67)
    X_train, y_train = create_dataset(train, WINDOW_SIZE)
    X_test, y_test = create_dataset(test, WINDOW_SIZE)

    # Construction
    model = build_lstm_model()
    model.summary()

    # Entraînement
    history = train_model(model, X_train, y_train, epochs=100, batch_size=8)

    # Évaluation + visualisations
    train_pred, test_pred, rmse_train, rmse_test = evaluate_model(
        model, X_train, y_train, X_test, y_test, scaler,
    )
    plot_loss(history)
    plot_predictions(series_scaled, train_pred, test_pred, WINDOW_SIZE, scaler)

    # Sauvegarde (utilisé éventuellement en phase 8 piste C)
    model.save("airline_lstm.keras")
    print("\nModèle Airline LSTM sauvegardé : airline_lstm.keras")

    # ===== Quality gates =====
    # Happy path : observer la courbe — la prédiction suit visuellement la
    #              tendance + saisonnalité, RMSE < 70.
    # Edge case : enlever EarlyStopping et passer epochs=5 → underfitting,
    #             val_loss encore décroissante en fin.
    # Adversarial : inverser train/test (entraîner sur test, évaluer sur train)
    #               → bonne perf trompeuse (le futur prédit le passé) =
    #               contamination temporelle / data leakage.
