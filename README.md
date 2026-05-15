# Projet J4 — RNN, LSTM, séries temporelles, déploiement

Livrable du jour 4 : pipeline complet RNN/LSTM/GRU sur séries temporelles
(Airline Passengers) et NLP (IMDB sentiment, AG News), suivi du déploiement
d'une WebApp Streamlit.

## Structure

```
.
├── README.md
├── requirements.txt
├── results.md                          # tableau comparatif phase 3
├── .gitignore
│
├── phase1_sliding_window.py            # série temporelle + fenêtre glissante
├── phase2_lstm_training.py             # LSTM + RMSE
├── phase3_gru_comparison.py            # benchmark LSTM vs GRU
│
├── phase4_tokenization.py              # IMDB tokenization + embedding
├── phase5_bidirectional_lstm.py        # Bi-LSTM + évaluation (génère imdb_sentiment.keras)
│
├── phase6_multiclass.py                # AG News, 4 classes
│
├── app.py                              # phase 7 — Streamlit basique
├── app_with_logging.py                 # phase 8 piste B — logging
└── phase8_stacked_lstm.py              # phase 8 piste D — LSTM stacké
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

## Exécution séquentielle

Chaque script est exécutable indépendamment depuis la racine du projet.

```bash
# --- Bloc séries temporelles ---
python phase1_sliding_window.py
python phase2_lstm_training.py         # sauvegarde airline_lstm.keras
python phase3_gru_comparison.py        # sauvegarde airline_gru.keras

# --- Bloc NLP ---
python phase4_tokenization.py
python phase5_bidirectional_lstm.py    # sauvegarde imdb_sentiment.keras
python phase6_multiclass.py            # télécharge AG News (~30 Mo) la 1ère fois

# --- WebApp ---
streamlit run app.py                   # nécessite imdb_sentiment.keras

# --- Phase 8 (bac à sable, choisir 1 ou 2 pistes) ---
streamlit run app_with_logging.py
```

## Commits suggérés (un par phase)

```
feat: phase1 sliding window airline passengers
feat: phase2 lstm airline passengers training
feat: phase3 gru vs lstm comparison airline passengers
feat: phase4 imdb tokenization embedding
feat: phase5 imdb bidirectional lstm training
feat: phase6 ag_news multiclass lstm
feat: phase7 streamlit webapp imdb inference
feat: phase8 inference logging csv
```

## Notes importantes

- Le modèle `imdb_sentiment.keras` (~5–10 Mo) est produit par la phase 5 et doit
  être présent dans le même dossier que `app.py` pour que la WebApp fonctionne.
- La phase 6 télécharge AG News depuis HuggingFace (~30 Mo, mis en cache après
  le premier appel).
- Jamais de shuffle sur les séries temporelles : le split est strictement
  chronologique (`train_ratio=0.67`).
- Chaque script imprime ses propres quality gates (happy path, edge case,
  adversarial) à la fin de son exécution.

## Dépendances

Voir `requirements.txt`. Le projet utilise Keras 3 (TensorFlow backend).
