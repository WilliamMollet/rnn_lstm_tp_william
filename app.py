"""
Phase 7 — WebApp Streamlit : inférence IMDB sentiment.

Lancer avec :
    streamlit run app.py

Pré-requis :
    le fichier imdb_sentiment.keras (généré par phase5_bidirectional_lstm.py)
    doit être présent dans le même dossier.
"""
import os
import streamlit as st
import keras
from keras.datasets import imdb
from keras.utils import pad_sequences


VOCAB_SIZE = 10000
MAX_LEN = 200
MODEL_PATH = "imdb_sentiment.keras"


# ----- Chargement (caché : exécuté une seule fois par session Streamlit) -----

@st.cache_resource
def load_model():
    """@st.cache_resource garde l'objet en RAM entre les reruns."""
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Modèle introuvable : {MODEL_PATH}\n"
            f"Exécutez d'abord `python phase5_bidirectional_lstm.py` "
            f"pour générer le fichier."
        )
        st.stop()
    return keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_word_index():
    return imdb.get_word_index()


# ----- Pipeline d'inférence -----

def preprocess_text(text, word_index, max_len=MAX_LEN, vocab_size=VOCAB_SIZE):
    """
    Tokenise une review en respectant la convention IMDB :
      - 0 = padding, 1 = start, 2 = OOV, 3 = unused
      - les indices de word_index sont décalés de +3 quand load_data() encode
    """
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
    return pad_sequences(
        [tokens], maxlen=max_len, padding="pre", truncating="pre",
    )


# ----- UI Streamlit -----

def main():
    st.set_page_config(page_title="IMDB Sentiment", page_icon="🎬")
    st.title("🎬 Analyse de sentiment — Reviews IMDB")
    st.write(
        "Saisissez une review de film en **anglais**. Le modèle (LSTM "
        "bidirectionnel entraîné sur 25 000 reviews IMDB) prédira si le "
        "sentiment est positif ou négatif."
    )
    st.caption("Note : les textes de plus de 200 mots sont tronqués.")

    model = load_model()
    word_index = load_word_index()

    user_input = st.text_area(
        "Votre review :",
        height=150,
        placeholder="Type your movie review here...",
    )

    if st.button("🔍 Analyser"):
        # Quality gate — edge case : input vide
        if not user_input.strip():
            st.warning("Veuillez saisir une review avant d'analyser.")
        else:
            padded = preprocess_text(user_input, word_index)
            proba = float(model.predict(padded, verbose=0)[0, 0])

            if proba >= 0.5:
                st.success(
                    f"✅ Sentiment : **Positif** (confiance : {proba:.2%})"
                )
            else:
                st.error(
                    f"❌ Sentiment : **Négatif** (confiance : {1 - proba:.2%})"
                )

            st.caption(f"Score brut du modèle : `{proba:.4f}` (seuil 0.5)")

            with st.expander("Détails techniques"):
                n_tokens_reels = int((padded > 0).sum())
                st.write(f"Tokens non-padding : **{n_tokens_reels}**")
                st.write(f"Longueur de la séquence : **{MAX_LEN}**")
                st.write(f"Taille du vocabulaire : **{VOCAB_SIZE}**")

    # Sidebar : info modèle (rôle de pseudo-endpoint /health)
    with st.sidebar:
        st.header("À propos")
        st.write(
            "Cette WebApp utilise un LSTM bidirectionnel entraîné sur IMDB.\n\n"
            "**Architecture :**\n"
            "- Embedding (vocab=10 000, dim=128)\n"
            "- Bidirectional LSTM (64 units)\n"
            "- Dropout 0.5\n"
            "- Dense sigmoid\n\n"
            "**Performance :** ~87 % accuracy sur le test set IMDB."
        )
        st.divider()
        st.caption(f"Modèle : `{MODEL_PATH}`")


if __name__ == "__main__":
    main()
