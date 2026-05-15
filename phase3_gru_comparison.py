"""
Phase 3 — Mini-phase GRU : comparaison cross-architecture sur Airline Passengers.

Objectif : remplacer le LSTM de la phase 2 par un GRU, mesurer 5 métriques
(RMSE test, training time total, durée/epoch, nb de paramètres, val_loss
finale) et imprimer un tableau comparatif.
"""
import time
import numpy as np
import keras
from keras import layers
from sklearn.metrics import mean_squared_error

from phase1_sliding_window import (
    load_airline_data, normalize_series, temporal_split,
    create_dataset, WINDOW_SIZE,
)


def build_and_train(rnn_layer_class, X_train, y_train, X_test, y_test, scaler,
                    units=64, epochs=50, batch_size=8, label="", verbose=0):
    """
    Construit et entraîne un modèle avec le layer RNN passé en paramètre.

    rnn_layer_class peut être layers.LSTM, layers.GRU, layers.SimpleRNN, etc.

    Retourne un dict contenant le modèle, l'historique, et toutes les métriques.
    """
    print(f"\n=== Training {label} ===")

    model = keras.Sequential([
        rnn_layer_class(units, input_shape=(WINDOW_SIZE, 1)),
        layers.Dropout(0.2),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True,
    )

    start = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=verbose,
    )
    duration = time.time() - start

    # RMSE test (dénormalisé pour rester en unité passagers)
    test_pred = model.predict(X_test, verbose=0)
    test_pred_orig = scaler.inverse_transform(test_pred)
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse_test = float(np.sqrt(mean_squared_error(y_test_orig, test_pred_orig)))

    nb_params = model.count_params()
    real_epochs = len(history.history["loss"])
    val_loss_final = float(history.history["val_loss"][-1])

    print(f"{label} :")
    print(f"  Paramètres        : {nb_params:,}")
    print(f"  Epochs réelles    : {real_epochs}/{epochs}")
    print(f"  Durée totale      : {duration:.2f} s")
    print(f"  Durée / epoch     : {duration / real_epochs:.3f} s")
    print(f"  Val loss finale   : {val_loss_final:.5f}")
    print(f"  RMSE test         : {rmse_test:.2f} passagers")

    return {
        "label": label,
        "model": model,
        "history": history,
        "duration_total": duration,
        "duration_per_epoch": duration / real_epochs,
        "real_epochs": real_epochs,
        "rmse_test": rmse_test,
        "nb_params": nb_params,
        "val_loss_final": val_loss_final,
    }


def print_comparison_table(*results):
    """Imprime un tableau comparatif markdown-friendly."""
    print("\n" + "=" * 70)
    print("Tableau comparatif")
    print("=" * 70)

    header = f"{'Métrique':<25}" + "".join(f"{r['label']:>15}" for r in results)
    print(header)
    print("-" * 70)

    rows = [
        ("RMSE test",        "{:>15.2f}",  "rmse_test"),
        ("Training time (s)","{:>15.2f}",  "duration_total"),
        ("Durée/epoch (s)",  "{:>15.3f}",  "duration_per_epoch"),
        ("Nb paramètres",    "{:>15,}",    "nb_params"),
        ("Val loss finale",  "{:>15.5f}",  "val_loss_final"),
    ]
    for name, fmt, key in rows:
        row = f"{name:<25}" + "".join(fmt.format(r[key]) for r in results)
        print(row)
    print("=" * 70)


if __name__ == "__main__":
    # Données
    series = load_airline_data()
    series_scaled, scaler = normalize_series(series)
    train, test = temporal_split(series_scaled, train_ratio=0.67)
    X_train, y_train = create_dataset(train, WINDOW_SIZE)
    X_test, y_test = create_dataset(test, WINDOW_SIZE)

    # Benchmark LSTM vs GRU avec les mêmes hyperparamètres
    lstm_result = build_and_train(
        layers.LSTM, X_train, y_train, X_test, y_test, scaler,
        units=64, epochs=50, label="LSTM",
    )
    gru_result = build_and_train(
        layers.GRU, X_train, y_train, X_test, y_test, scaler,
        units=64, epochs=50, label="GRU",
    )
    print_comparison_table(lstm_result, gru_result)

    # Sauvegarde du GRU pour la phase 8 (piste C éventuelle)
    gru_result["model"].save("airline_gru.keras")
    print("\nModèle GRU sauvegardé : airline_gru.keras")

    # ===== Quality gates =====

    # Edge case : doubler les epochs du GRU
    print("\n--- Edge case : GRU avec 100 epochs ---")
    gru_long = build_and_train(
        layers.GRU, X_train, y_train, X_test, y_test, scaler,
        units=64, epochs=100, label="GRU (100ep)",
    )
    print_comparison_table(lstm_result, gru_result, gru_long)

    # Adversarial : bruit blanc — les deux modèles doivent échouer
    print("\n--- Adversarial : bruit blanc pur ---")
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(144).reshape(-1, 1).astype("float32")
    noise_scaled, noise_scaler = normalize_series(noise)
    noise_train, noise_test = temporal_split(noise_scaled, 0.67)
    Xn_tr, yn_tr = create_dataset(noise_train)
    Xn_te, yn_te = create_dataset(noise_test)

    noise_lstm = build_and_train(
        layers.LSTM, Xn_tr, yn_tr, Xn_te, yn_te, noise_scaler,
        units=64, epochs=30, label="LSTM/bruit",
    )
    noise_gru = build_and_train(
        layers.GRU, Xn_tr, yn_tr, Xn_te, yn_te, noise_scaler,
        units=64, epochs=30, label="GRU/bruit",
    )
    print_comparison_table(noise_lstm, noise_gru)
    print(
        "\nLecture : un RMSE 'bon' sur du bruit signalerait un overfit du train. "
        "Sur du vrai bruit blanc, le RMSE doit refléter la variance naturelle."
    )
