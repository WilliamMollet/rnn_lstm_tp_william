"""
Phase 4 — TP NLP IMDB : tokenization et embedding.

Objectif : préparer le dataset IMDB Sentiment pour un LSTM. Comprendre la
tokenization (déjà fournie par Keras), le padding, et le rôle de l'Embedding.
"""
import numpy as np
import keras
from keras import layers
from keras.datasets import imdb

# Keras 3 expose pad_sequences via keras.utils (et conserve l'ancien chemin
# keras.preprocessing.sequence.pad_sequences en alias).
from keras.utils import pad_sequences


VOCAB_SIZE = 10000   # garde le top 10 000 mots les plus fréquents
MAX_LEN = 200        # longueur fixe après padding
EMBED_DIM = 128      # dimension du vecteur dense par mot


def load_imdb_data(vocab_size=VOCAB_SIZE):
    """Charge IMDB : 25 000 reviews train + 25 000 reviews test, déjà tokenisées."""
    (X_train_raw, y_train), (X_test_raw, y_test) = imdb.load_data(
        num_words=vocab_size,
    )
    print(f"Reviews train : {len(X_train_raw)}, reviews test : {len(X_test_raw)}")

    lengths = [len(x) for x in X_train_raw]
    print(
        f"Longueurs train — min: {min(lengths)}, max: {max(lengths)}, "
        f"moyenne: {np.mean(lengths):.1f}, médiane: {np.median(lengths):.1f}"
    )
    print(
        f"Distribution labels : {np.bincount(y_train)} (0=Négatif, 1=Positif)"
    )
    return X_train_raw, y_train, X_test_raw, y_test


def pad_data(X_train_raw, X_test_raw, max_len=MAX_LEN, padding="pre"):
    """
    Padding 'pre' (zéros au début) vs 'post' (zéros à la fin).

    'pre' est généralement préférable pour un LSTM many-to-one : les zéros
    arrivent d'abord, puis le contenu réel arrive en fin de séquence, donc
    le hidden state final est dominé par les vrais tokens, pas par le padding.
    """
    X_train_pad = pad_sequences(
        X_train_raw, maxlen=max_len, padding=padding, truncating=padding,
    )
    X_test_pad = pad_sequences(
        X_test_raw, maxlen=max_len, padding=padding, truncating=padding,
    )
    return X_train_pad, X_test_pad


def build_embedding_demo(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
                          max_len=MAX_LEN):
    """
    Petit modèle pour visualiser la transformation Embedding.

    Input  shape : (None, MAX_LEN)               -- séquence d'indices entiers
    Output shape : (None, MAX_LEN, EMBED_DIM)    -- séquence de vecteurs denses
    Params       : VOCAB_SIZE * EMBED_DIM = 1 280 000
    """
    model = keras.Sequential([
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            input_length=max_len,
        ),
    ])
    model.build(input_shape=(None, max_len))
    return model


if __name__ == "__main__":
    X_train_raw, y_train, X_test_raw, y_test = load_imdb_data()
    X_train_pad, X_test_pad = pad_data(X_train_raw, X_test_raw, padding="pre")

    print(f"\nX_train_pad shape : {X_train_pad.shape}")  # attendu : (25000, 200)
    print(f"X_test_pad shape  : {X_test_pad.shape}")

    print("\n--- Embedding demo ---")
    embed_model = build_embedding_demo()
    embed_model.summary()

    # ===== Quality gates =====

    # Edge case : reviews très courtes (< 10 tokens)
    print("\n--- Edge case : reviews < 10 tokens ---")
    short_indices = [i for i, x in enumerate(X_train_raw) if len(x) < 10]
    print(f"Nombre de reviews < 10 tokens : {len(short_indices)}")
    if short_indices:
        idx = short_indices[0]
        original = X_train_raw[idx]
        padded = X_train_pad[idx]
        print(f"Review #{idx} (longueur originale {len(original)}) :")
        print(f"  Tokens originaux : {original}")
        print(f"  Padding pre — 20 premiers : {padded[:20]}")
        print(f"  Padding pre — 20 derniers : {padded[-20:]}")
        print(
            "Note : avec padding='pre', les vrais tokens sont en fin de séquence, "
            "donc le hidden state final du LSTM les voit en dernier."
        )

    # Adversarial : vocabulaire trop petit (num_words=100)
    print("\n--- Adversarial : vocab_size=100 ---")
    (X_tiny, _), _ = imdb.load_data(num_words=100)
    # Compter les tokens OOV (index 2 quand index_from=3)
    n_oov = sum((np.array(x) == 2).sum() for x in X_tiny[:1000])
    n_total = sum(len(x) for x in X_tiny[:1000])
    print(
        f"Sur 1000 reviews, {n_oov}/{n_total} tokens sont OOV "
        f"({100 * n_oov / n_total:.1f} %)."
    )
    print(
        "Avec un vocab trop petit, presque tous les mots deviennent <OOV>, "
        "le modèle perd toute information sémantique → accuracy ~50 % (random)."
    )
