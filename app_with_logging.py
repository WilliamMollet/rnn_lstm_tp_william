"""
Phase 8 — Piste B : WebApp avec logging d'inférence en CSV.

Lancer avec :
    streamlit run app_with_logging.py

Améliorations par rapport à app.py (phase 7) :
  - Chaque inférence est loggée dans inference_log.csv (timestamp, input
    tronqué, prediction, confiance).
  - Un panneau "Monitoring" affiche les 10 dernières inférences.
  - La sidebar joue le rôle d'un endpoint /health (état du modèle, taille
    du fichier de log).
"""
import csv
import datetime
import os
import streamlit as st
import keras
from keras.datasets import imdb
from keras.utils import pad_sequences


VOCAB_SIZE = 10000
MAX_LEN = 200
MODEL_PATH = "imdb_sentiment.keras"
LOG_FILE = "inference_log.csv"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Modèle introuvable : {MODEL_PATH}\n"
            f"Exécutez d'abord `python phase5_bidirectional_lstm.py`."
        )
        st.stop()
    return keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_word_index():
    return imdb.get_word_index()


def preprocess_text(text, word_index, max_len=MAX_LEN, vocab_size=VOCAB_SIZE):
    tokens = []
    for word in text.lower().split():
        word_clean = "".join(c for c in word if c.isalnum())
        if not word_clean:
            continue
        idx = word_index.get(word_clean, 0)
        if idx == 0 or idx + 3 >= vocab_size:
            tokens.append(2)
        else:
            tokens.append(idx + 3)
    if not tokens:
        tokens = [2]
    return pad_sequences(
        [tokens], maxlen=max_len, padding="pre", truncating="pre",
    )


def log_inference(input_text, prediction, confidence, log_file=LOG_FILE):
    """
    Loggue chaque inférence dans un CSV local.

    En production réelle ce serait une base de données ou un service de
    monitoring type Prometheus / Sentry. Ici on reste sur un CSV pour
    rester transparent.
    """
    file_exists = os.path.exists(log_file)
    # csv.writer gère l'échappement des virgules et guillemets. On nettoie
    # quand même les retours ligne car ils cassent la lecture humaine.
    input_clean = input_text.replace("\n", " ").replace("\r", " ")[:100]
    with open(log_file, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "input", "prediction", "confidence"])
        writer.writerow([
            datetime.datetime.now().isoformat(timespec="seconds"),
            input_clean,
            prediction,
            f"{confidence:.4f}",
        ])


def read_recent_logs(n=10, log_file=LOG_FILE):
    """Renvoie les n dernières lignes du log (plus récentes en premier)."""
    if not os.path.exists(log_file):
        return []
    with open(log_file, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) <= 1:  # juste l'en-tête, pas de données
        return []
    return rows[-n:][::-1]


def main():
    st.set_page_config(
        page_title="IMDB Sentiment (logged)", page_icon="🎬",
    )
    st.title("🎬 Analyse de sentiment — Reviews IMDB (avec monitoring)")
    st.caption("Note : les textes de plus de 200 mots sont tronqués.")

    model = load_model()
    word_index = load_word_index()

    user_input = st.text_area("Votre review :", height=150)

    if st.button("🔍 Analyser"):
        if not user_input.strip():
            st.warning("Veuillez saisir une review avant d'analyser.")
        else:
            padded = preprocess_text(user_input, word_index)
            proba = float(model.predict(padded, verbose=0)[0, 0])

            if proba >= 0.5:
                label = "Positif"
                confidence = proba
                st.success(
                    f"✅ Sentiment : **Positif** (confiance : {confidence:.2%})"
                )
            else:
                label = "Négatif"
                confidence = 1 - proba
                st.error(
                    f"❌ Sentiment : **Négatif** (confiance : {confidence:.2%})"
                )

            # Logging
            log_inference(user_input, label, confidence)
            st.caption(f"Inférence loggée dans `{LOG_FILE}`")

    # Panneau monitoring
    with st.expander("📊 Monitoring : 10 dernières inférences"):
        recent = read_recent_logs(10)
        if recent:
            st.table([
                {
                    "Timestamp": r[0],
                    "Input": r[1],
                    "Prediction": r[2],
                    "Confidence": r[3],
                }
                for r in recent
            ])
        else:
            st.info("Aucune inférence loggée pour l'instant.")

    # Sidebar — pseudo-endpoint /health
    with st.sidebar:
        st.header("🔧 Health")
        st.write(f"**Modèle :** `{MODEL_PATH}`")
        st.write(f"**Vocab size :** {VOCAB_SIZE}")
        st.write(f"**Max length :** {MAX_LEN}")
        st.write(f"**Log file :** `{LOG_FILE}`")
        if os.path.exists(LOG_FILE):
            size = os.path.getsize(LOG_FILE)
            # Compter les lignes (- 1 pour l'en-tête)
            with open(LOG_FILE, encoding="utf-8") as f:
                n_rows = sum(1 for _ in f) - 1
            st.write(f"**Log size :** {size} bytes")
            st.write(f"**Total inférences :** {max(0, n_rows)}")
        else:
            st.write("**Log size :** 0 (pas encore d'inférence)")


if __name__ == "__main__":
    main()
