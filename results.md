# Résultats — Phase 3 : comparaison LSTM vs GRU sur Airline Passengers

**Dataset :** Airline Passengers (144 valeurs mensuelles, 1949-1960)
**Fenêtre glissante :** 12 mois → prédire le 13e
**Split temporel :** 67 % train / 33 % test (jamais de shuffle)
**Architecture :** RNN(64 units) → Dropout(0.2) → Dense(1)
**Loss :** MSE, **optimizer :** Adam, **métrique humaine :** MAE
**Callback :** EarlyStopping(patience=10, restore_best_weights=True)

## Tableau comparatif

| Métrique               | LSTM             | GRU              |
| ----------------------- | ---------------- | ---------------- |
| RMSE test (passagers)   | _1.18_         | _1.16_         |
| Training time total (s) | _2.52_         | _3.09_         |
| Durée par epoch (s)    | _0.229_        | _0.221_        |
| Nombre de paramètres   | **16 961** | **12 929** |
| Val loss finale         | _0.02668_      | _0.02573_      |

## Observations attendues

- GRU a ~25 % moins de paramètres que LSTM (2 gates au lieu de 4 sous-opérations).
- Sur ce dataset (très petit, 84 samples d'entraînement), la différence de RMSE
  est marginale (< 5 passagers).
- GRU est légèrement plus rapide par epoch.
- Sur des datasets plus larges ou des séquences plus longues, l'écart se creuse.

## Edge case : doubler les epochs GRU

Lancer GRU à 100 epochs vs LSTM à 50 : est-ce que plus d'epochs GRU rattrape la
performance LSTM ? Souvent oui sur ce dataset (la convergence est plus lente
mais le plateau est similaire).

## Adversarial : bruit blanc

Entraîner LSTM et GRU sur `np.random.randn(144, 1)` : le RMSE doit être mauvais
pour les deux (une série aléatoire est imprévisible par construction). Si un
modèle obtient un "bon" RMSE sur du bruit, c'est qu'il overfit le training set.

## Conclusion

Pour ce type de série temporelle courte avec saisonnalité simple, GRU et LSTM
donnent des performances équivalentes. **GRU est préférable par défaut** :
moins de paramètres, entraînement plus rapide, moins de risque d'overfit.
LSTM devient pertinent sur des séquences longues (> 200 steps) avec des
dépendances à long terme.
