# Classification-d-un-segment-de-discours-politique
Ce projet consiste à développer un modèle de deeplearning capable d'identifier le locuteur (Chirac vs. Mitterrand) d'un segment de discours politique.

# Dataset
Les données que nous utilisons proviennent de : https://deft.limsi.fr/, ce jeu de données contient au total 54017 segments de discours répartis comme suit : ![alt text](https://github.com/dahmri/Classification-d-un-segment-de-discours-politique/blob/master/Figures/Nombre%20de%20textes.png)

# Modèle

- CNN mono couche avec les hyperparamètres suivants : filter_sizes = 3, num_filters = 100, GlobalMaxPooling,
- Embedding_dim = Word2Vec 100
- Dropout = 0.3
- Activation function = "sigmoid"

# Tests et Résultats
Pour nos tests nous avons laissé de côté 2000 textes (1000 pour chaque locuteur), et nous avons utilisé le reste de données pour entrainement (80%) et validation (20%) du modèle.

![alt text](https://github.com/dahmri/Classification-d-un-segment-de-discours-politique/blob/master/Figures/Distribution%20de%20la%20pre%CC%81diction.png)

![alt text](https://github.com/dahmri/Classification-d-un-segment-de-discours-politique/blob/master/Figures/matrice%20de%20confusion.png)

![alt text](https://github.com/dahmri/Classification-d-un-segment-de-discours-politique/blob/master/Figures/ROC%20Curve.png)

