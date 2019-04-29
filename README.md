# Classification-d-un-segment-de-discours-politique
Ce projet consiste à développer un modèle de deeplearning capable d'identifier le locuteur (Chirac vs. Mitterrand) d'un segment de discours politique.

# Dataset
Les données que nous utilisons proviennent de : https://deft.limsi.fr/, ce jeu de données contient au total 54017 segments de discours répartis comme suit : ![alt text](https://github.com/dahmri/Classification-d-un-segment-de-discours-politique/blob/master/Figures/Nombre%20de%20textes.png)

# Modèle

CNN mono couche avec les hyperparamètres suivants :   
* filter_sizes = 3
* num_filters = 100
* GlobalMaxPooling
embedding_dim = 100
Dropout= 0.3

# Résultats
