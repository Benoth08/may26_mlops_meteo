# Surveillance et réentraînement

## Les signaux, du plus précoce au plus fiable

| Signal | Ce qu'il observe | Vérité terrain nécessaire | Délai | Où |
|---|---|---|---|---|
| Qualité des données | Doublons, hausse des valeurs manquantes | non | immédiat | `weather_drift` |
| Dérive des données | Distribution des variables d'entrée | non | hebdomadaire | `weather_drift` |
| Dérive des prédictions | Distribution des sorties du modèle | non | hebdomadaire | `weather_drift` |
| Chute de performance | F1 sous un seuil | **oui** | au réentraînement | voir plus bas |

### Pourquoi plusieurs signaux

En production, on ne sait pas s'il a plu avant le lendemain. Le F1 ne peut
donc pas être calculé en temps réel : c'est le signal le plus sûr, mais le
plus tardif.

La dérive des prédictions comble ce trou. Si le modèle annonce de la pluie
38 % du temps alors qu'il était à 25 %, on le sait immédiatement, sans aucune
étiquette.

La dérive des données détecte le changement en amont, avant même que les
prédictions ne bougent.

La qualité ne mesure pas le monde mais le pipeline : des doublons ou une
hausse de valeurs manquantes signalent une panne technique.

### État du quatrième signal

La chute de performance n'est **pas encore branchée**. La variable Airflow
`F1_ALERT_THRESHOLD` est transmise par `weather_models` mais aucun code ne la
lit aujourd'hui. Le garde-fou réellement actif au moment de la promotion est
`min_score_improvement`, décrit plus bas.

## Comparaison à un an d'écart

La fenêtre récente est comparée à la même période un an plus tôt, et non aux
mois précédents.

Sans cela, l'écart entre l'été et l'hiver serait signalé comme une dérive à
chaque exécution. Sur des données météo australiennes, c'est garanti.

Vérifié par simulation sur des données purement saisonnières :

| Référence | Part de colonnes dérivées |
|---|---|
| Tout l'historique antérieur | 1,00 — 100 % de fausses alertes |
| Même période un an plus tôt | 0,00 |
| Un an plus tôt, avec dérive réelle injectée | 0,67 — détectée |

Les deux fenêtres sont ancrées sur le `MAX(date)` de `weather_data_raw`, pas
sur la date du jour. La surveillance fonctionne donc sur un jeu historique,
mais elle exige au moins 400 jours de données contiguës en base : 30 jours
pour la fenêtre courante, 365 de recul, 30 pour la fenêtre de référence.

## Seuils calibrés par la mesure

Le seuil par défaut d'Evidently, 0,10 par colonne, est trop sensible pour ces
données : le premier essai signalait 21 colonnes sur 23.

Méthode retenue : mesurer le bruit de fond avant de fixer le seuil. Une même
période de 60 jours est coupée en deux au hasard. Les deux moitiés sont
homogènes par construction, donc tout ce qui ressort est du bruit
d'échantillonnage.

| Seuil par colonne | Bruit de fond mesuré | Années consécutives réelles |
|---|---|---|
| 0,10 (défaut Evidently) | 0,50 / 0,50 / 0,50 | 0,95 / 0,68 / 0,50 / 0,68 / 0,64 |
| 0,20 | 0,32 / 0,36 / 0,32 | 0,41 / 0,41 / 0,32 / 0,36 / 0,36 |
| **0,30 (retenu)** | **0,00 / 0,05 / 0,00** | **0,14 / 0,14 / 0,09 / 0,00 / 0,09** |

À 0,10, le bruit couvre la moitié des colonnes : aucun seuil global n'est
exploitable. À 0,30, le bruit s'effondre et le signal réel subsiste.

Réglage retenu, dans `config/variables.json` :

| Variable | Valeur | Rôle |
|---|---|---|
| `DRIFT_COLUMN_THRESHOLD` | 0.30 | Distance au-delà de laquelle une colonne est dite dérivée |
| `DRIFT_SHARE_THRESHOLD` | 0.10 | Part de colonnes dérivées déclenchant l'alerte |
| `DRIFT_WINDOW_DAYS` | 30 | Largeur des deux fenêtres comparées |
| `QUALITY_MISSING_INCREASE` | 0.10 | Hausse de valeurs manquantes jugée anormale |

## Méthode statistique figée

Evidently change automatiquement de test statistique en dessous de 1 000
lignes. La surveillance changerait donc de comportement à mesure que les
données s'accumulent — inacceptable pour du monitoring.

La méthode est donc imposée : **Wasserstein** pour les variables numériques,
**Jensen-Shannon** pour les catégorielles.

## Colonnes exclues

`date_import`, `run_id` et `data_source` changent à chaque import et seraient
systématiquement signalées comme dérivées. `date` et `id` sont techniques.
Toutes les cinq sont écartées de la comparaison.

## Règle de décision

```
qualité dégradée                   -> QUALITE : alerte, PAS de réentraînement
dérive des données ou prédictions  -> DERIVE  : alerte
sinon                              -> OK      : rien
```

La qualité prime volontairement sur la dérive. Des doublons ou une hausse de
valeurs manquantes signalent un pipeline cassé, pas un monde qui change :
réentraîner dans ce cas reviendrait à apprendre la corruption.

**La tâche d'alerte échoue délibérément** quand une dérive ou un problème de
qualité est détecté. C'est un choix : sans pile d'alerting, l'échec de tâche
est le seul moyen de rendre le signal visible dans Airflow — un DAG vert ne
doit pas masquer une alerte.

Conséquence à connaître : une exécution de `weather_drift` en rouge peut
vouloir dire deux choses opposées, soit que la surveillance a détecté quelque
chose, soit qu'elle a planté. Il faut lire les journaux de `detect_drift` pour
trancher, la décision y figure en dernière ligne. Brancher le signal sur les
alertes Prometheus lèverait cette ambiguïté.

## Réentraînement

Deux déclencheurs, et un troisième qui n'existe pas encore.

| Déclencheur | État | Mécanisme |
|---|---|---|
| Arrivée de données | actif | `weather_training_trigger`, chaque heure : 1 000 nouvelles lignes ou 24 h écoulées, puis `TriggerDagRunOperator` vers `weather_models` |
| Cadence | actif | `weather_models` tourne le 1er du mois à 3 h, en filet de sécurité |
| Dérive détectée | **non branché** | `weather_drift` alerte, il ne déclenche rien |

Brancher la dérive sur le réentraînement est la prochaine étape. Cela n'a pas
été fait tant que les seuils n'étaient pas calés sur des mesures réelles.

Les seuils de déclenchement sont dans `params.yml`, section `training`, donc
versionnés par DVC : on sait avec quelle règle chaque modèle a été produit.

## Promotion

Un modèle fraîchement entraîné n'est pas un modèle de production. Il ne le
devient que s'il bat celui en place.

Point essentiel : les deux sont notés sur le **même jeu de test**. Le modèle en
production est rechargé depuis le Registry MLflow et réévalué sur les données
du moment, celles-là mêmes qui ont servi à noter le candidat.

Sans cela, on comparerait son score archivé — calculé sur un jeu de test plus
ancien — au score du nouveau modèle. Deux notes obtenues à deux épreuves
différentes. Avec des données qui arrivent en continu, ce cas se produit à
chaque cycle, et rien ne le signale : les deux nombres portent le même nom et
la même échelle.

Deux garde-fous :

- `min_score_improvement`, 0,01 par défaut : un gain inférieur à 1 % est tenu
  pour du bruit et ne donne pas lieu à promotion ;
- si le modèle en production ne peut pas être rechargé, **aucune promotion**.
  Mieux vaut ne rien faire que promouvoir sans comparaison.

Conséquence : le réentraînement automatique ne peut pas dégrader la
production, même s'il est déclenché par une fausse alerte.

## Limites connues

- Le seuil de dérive des prédictions réutilise 0,30, calibré sur des colonnes
  météo et non sur des probabilités. À recalibrer après les premières
  exécutions réelles.
- La calibration repose sur cinq paires d'années consécutives. C'est peu pour
  établir solidement la séparation entre bruit et signal.
- Le quatrième signal, la chute de performance, n'est pas implémenté.
- La dérive ne déclenche pas le réentraînement.
