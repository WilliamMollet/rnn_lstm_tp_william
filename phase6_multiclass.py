"""
Phase 6 — TP AG News : classification multiclasse NLP.

Objectif : classifier des titres d'articles de presse en 4 catégories
(World, Sports, Business, Sci/Tech) avec un LSTM.

Différences avec IMDB :
  - 4 classes au lieu de 2 → Dense(4, softmax) au lieu de Dense(1, sigmoid)
  - labels entiers 0..3 → sparse_categorical_crossentropy
  - vocabulaire non pré-tokenisé → on construit notre Tokenizer Keras
"""
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from sklearn.metrics import classification_report
from datasets import load_dataset


VOCAB_SIZE_AG = 20000
MAX_LEN_AG = 50       # les titres sont courts (< 30 mots), 50 suffit
EMBED_DIM_AG = 64
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def load_ag_news():
    """Télécharge AG News depuis HuggingFace (cache local après 1er appel)."""
    dataset = load_dataset("ag_news")
    train_texts = dataset["train"]["text"]
    train_labels = np.array(dataset["train"]["label"])
    test_texts = dataset["test"]["text"]
    test_labels = np.array(dataset["test"]["label"])

    print(f"Train : {len(train_texts)} exemples")
    print(f"Test  : {len(test_texts)} exemples")
    print(f"Classes : {sorted(set(train_labels))}")
    print(f"Distribution train : {np.bincount(train_labels)}")
    print(f"Exemple : {train_texts[0][:120]}...")
    print(f"Label   : {train_labels[0]} ({LABEL_NAMES[train_labels[0]]})")

    return train_texts, train_labels, test_texts, test_labels


def build_tokenizer_ag(train_texts, vocab_size=VOCAB_SIZE_AG):
    """
    IMPORTANT : on fit le tokenizer UNIQUEMENT sur le train.

    Fit sur le test entraîne du data leakage : on incluerait dans le vocab
    des mots qu'on ne devrait pas connaître à l'entraînement.
    """
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)
    return tokenizer


def encode_texts(tokenizer, texts, max_len=MAX_LEN_AG):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(
        sequences, maxlen=max_len, padding="post", truncating="post",
    )


def build_ag_model(vocab_size=VOCAB_SIZE_AG, embed_dim=EMBED_DIM_AG,
                   max_len=MAX_LEN_AG, lstm_units=64, dropout=0.3):
    """
    Embedding → LSTM → Dropout → Dense(4, softmax)

    Softmax : transforme 4 logits en 4 probabilités qui somment à 1.
    """
    model = keras.Sequential([
        layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim, input_length=max_len,
        ),
        layers.LSTM(lstm_units),
        layers.Dropout(dropout),
        layers.Dense(4, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        # Labels entiers 0/1/2/3 → sparse_categorical (pas besoin de one-hot)
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def predict_ag_news(model, tokenizer, text,
                    max_len=MAX_LEN_AG, labels=LABEL_NAMES):
    """Tokenize, prédit, affiche les probabilités par classe."""
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(
        seq, maxlen=max_len, padding="post", truncating="post",
    )
    probas = model.predict(padded, verbose=0)[0]
    pred_idx = int(np.argmax(probas))
    pred_label = labels[pred_idx]

    print(f"\nTexte : {text}")
    for label, p in zip(labels, probas):
        marker = "  ←" if label == pred_label else ""
        print(f"  {label:<10} : {p:.4f}{marker}")
    print(f"  → Classe prédite : {pred_label}")
    return pred_label, probas


if __name__ == "__main__":
    # Données
    train_texts, train_labels, test_texts, test_labels = load_ag_news()

    # Tokenization (fit sur train seulement)
    tokenizer = build_tokenizer_ag(train_texts)
    X_train = encode_texts(tokenizer, train_texts)
    X_test = encode_texts(tokenizer, test_texts)

    print(f"\nX_train shape : {X_train.shape}")  # attendu : (120000, 50)
    print(f"X_test shape  : {X_test.shape}")

    # Modèle
    model = build_ag_model()
    model.summary()

    # Entraînement (3 epochs suffisent sur ce dataset relativement simple)
    history = model.fit(
        X_train, train_labels,
        epochs=3,
        batch_size=256,
        validation_split=0.05,
        verbose=1,
    )

    # Évaluation
    loss_ag, acc_ag = model.evaluate(X_test, test_labels, verbose=0)
    print(f"\nAccuracy AG News test : {acc_ag:.4f}  (objectif : > 0.85)")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("\nRapport de classification :")
    print(classification_report(
        test_labels, y_pred, target_names=LABEL_NAMES,
    ))

    # Sauvegarde modèle + tokenizer (sérialisé via pickle)
    model.save("ag_news_lstm.keras")
    with open("ag_news_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print("\nSauvegardés : ag_news_lstm.keras + ag_news_tokenizer.pkl")

    # ===== Quality gates =====

    # Adversarial : titres ambigus à cheval entre 2 catégories
    print("\n--- Adversarial : titres ambigus ---")
    ambiguous = [
        "Tesla announces record quarterly profits",
        "Olympics 2024 Tech companies sponsor all major sports events",
        "Apple unveils new health monitoring features in latest watch",
        "China beats Japan in football world cup qualifier",
        "Microsoft AI breakthrough changes how doctors diagnose cancer",
    ]
    for text in ambiguous:
        predict_ag_news(model, tokenizer, text)
