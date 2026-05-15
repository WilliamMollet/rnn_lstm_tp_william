"""
Phase 5 — TP NLP IMDB : LSTM bidirectionnel et évaluation.

Objectif : construire et entraîner un LSTM bidirectionnel pour la classification
de sentiment IMDB. Évaluer avec accuracy et F1-score. Sauvegarder le modèle
pour la WebApp de la phase 7.
"""
import numpy as np
import keras
from keras import layers
from keras.utils import pad_sequences
from sklearn.metrics import f1_score, classification_report

from phase4_tokenization import (
    load_imdb_data, pad_data, VOCAB_SIZE, MAX_LEN, EMBED_DIM,
)


def build_bidirectional_lstm(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
                              max_len=MAX_LEN, lstm_units=64, dropout=0.5):
    """
    Architecture :
      Embedding → Bidirectional(LSTM) → Dropout → Dense(1, sigmoid)

    Le wrapper Bidirectional crée deux LSTM (forward + backward) dont les
    hidden states sont concaténés. Sortie : 2 × lstm_units neurones.

    Dropout 0.5 : IMDB est bruité, on régularise fort.
    """
    model = keras.Sequential([
        layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim, input_length=max_len,
        ),
        layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False)),
        layers.Dropout(dropout),
        layers.Dense(1, activation="sigmoid"),  # binaire : proba ∈ [0, 1]
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",   # adaptée à la classification binaire
        metrics=["accuracy"],
    )
    return model


def train_imdb(model, X_train, y_train, epochs=5, batch_size=128):
    """3-5 epochs suffisent généralement sur IMDB avec un Bi-LSTM."""
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=3, restore_best_weights=True,
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


def evaluate_imdb(model, X_test, y_test):
    loss_test, acc_test = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nAccuracy test : {acc_test:.4f}  (objectif : > 0.85)")
    print(f"Loss test     : {loss_test:.4f}")

    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba >= 0.5).astype("int32")

    f1 = f1_score(y_test, y_pred)
    print(f"F1-score test : {f1:.4f}")

    print("\nRapport de classification :")
    print(classification_report(
        y_test, y_pred, target_names=["Négatif", "Positif"]
    ))
    return acc_test, f1, y_pred, y_pred_proba


def predict_custom_review(model, text, word_index=None,
                           max_len=MAX_LEN, vocab_size=VOCAB_SIZE):
    """
    Tokenize une review en respectant la convention IMDB :
      - 0 = padding
      - 1 = start
      - 2 = OOV / unknown
      - 3 = unused
    Et `imdb.load_data()` décale tous les indices de +3 par rapport au
    word_index renvoyé par `imdb.get_word_index()`. On doit reproduire ce
    décalage côté inference.
    """
    if word_index is None:
        word_index = keras.datasets.imdb.get_word_index()

    tokens = []
    for word in text.lower().split():
        word_clean = "".join(c for c in word if c.isalnum())
        if not word_clean:
            continue
        idx = word_index.get(word_clean, 0)
        if idx == 0 or idx + 3 >= vocab_size:
            tokens.append(2)  # OOV
        else:
            tokens.append(idx + 3)

    if not tokens:
        tokens = [2]

    padded = pad_sequences(
        [tokens], maxlen=max_len, padding="pre", truncating="pre",
    )
    proba = float(model.predict(padded, verbose=0)[0, 0])
    label = "Positif" if proba >= 0.5 else "Négatif"
    return label, proba


if __name__ == "__main__":
    # Données
    X_train_raw, y_train, X_test_raw, y_test = load_imdb_data()
    X_train_pad, X_test_pad = pad_data(X_train_raw, X_test_raw, padding="pre")

    # Modèle
    model = build_bidirectional_lstm()
    model.summary()

    # Entraînement
    history = train_imdb(model, X_train_pad, y_train, epochs=5, batch_size=128)

    # Évaluation
    acc, f1, y_pred, y_pred_proba = evaluate_imdb(
        model, X_test_pad, y_test,
    )

    # Checkpoint obligatoire avant la WebApp
    model.save("imdb_sentiment.keras")
    print("\nModèle IMDB sauvegardé : imdb_sentiment.keras")

    # ===== Quality gates =====

    # Adversarial : reviews custom (sarcasme, négation, etc.)
    print("\n--- Adversarial : reviews custom ---")
    word_index = keras.datasets.imdb.get_word_index()
    samples = [
        "Absolutely terrible boring beyond belief waste of time",
        "An amazing masterpiece I loved every single minute",
        "Really brilliant film perfect for falling asleep",       # sarcasme
        "A masterpiece of boredom I recommend it to those who want to suffer",
        "The acting was good but the script was a complete disaster",
    ]
    for text in samples:
        label, proba = predict_custom_review(model, text, word_index)
        print(f"  [{label} ({proba:.3f})]  {text}")

    # Edge case : reviews très courtes du test set
    print("\n--- Edge case : accuracy sur reviews < 10 tokens ---")
    short_idx = [i for i, x in enumerate(X_test_raw) if len(x) < 10]
    print(f"Nombre de reviews courtes dans le test : {len(short_idx)}")
    if short_idx:
        X_short = X_test_pad[short_idx]
        y_short = y_test[short_idx]
        _, acc_short = model.evaluate(X_short, y_short, verbose=0)
        print(f"Accuracy reviews courtes : {acc_short:.4f}")
        print(f"Accuracy globale         : {acc:.4f}")
        print(
            "Les reviews courtes contiennent peu d'information : on s'attend "
            "à une accuracy plus faible (moins de contexte pour le LSTM)."
        )
