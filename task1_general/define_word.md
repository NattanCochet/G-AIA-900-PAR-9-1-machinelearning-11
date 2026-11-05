# 2.1 General ML knowledge

## Overfitting
L'overfitting, c'est quand l'IA apprend par cœur ses données d'entraînement au lieu de comprendre la logique derrière. Résultat : elle a un excellent score sur les données avec lesquelles elle s'est entraînée, mais elle fait beaucoup d'erreurs sur les données de test, qu'elle n'a jamais vues.

Exemple : un élève apprend par cœur les corrigés de 10 exercices. Sur ces 10 exercices, il a 20/20. Le jour de l'examen, les exercices sont différents et il est perdu, parce qu'il a retenu les réponses et pas la méthode.

## Hyperparamètre
Un hyperparamètre, c'est un réglage de l'IA que l'on choisit avant de lancer l'entraînement. L'IA ne l'apprend pas elle-même, contrairement aux paramètres, qui sont ce qu'elle apprend. On peut le régler à la main, ou laisser un outil comme Optuna tester plusieurs valeurs et garder celle qui marche le mieux sur des données de validation.

Exemple : pour faire un gâteau, on règle la température du four et le temps de cuisson avant de commencer. Pour une IA, ça peut être le learning rate, le nombre de neurones ou le degré d'un polynôme.

## Features / labels
Les features sont les données qu'on donne à l'IA pour qu'elle réfléchisse. Le label est la bonne réponse qu'elle doit trouver. Pendant l'entraînement, on connaît les labels. Sur un nouvel exemple, c'est justement ce que l'IA doit deviner.

Exemple : deviner le prix d'une bouteille de vin. L'année et l'état de la bouteille sont les features, le prix correct est le label.

## Gradient et descente de gradient
La loss mesure à quel point le modèle se trompe : plus elle est basse, mieux c'est. Le gradient est la pente de la loss : il montre dans quelle direction l'erreur augmente. La descente de gradient, c'est faire des petits pas dans l'autre sens pour que l'erreur baisse, jusqu'à arriver au point le plus bas (le minimum d'erreur).

La taille des pas s'appelle le learning rate. Avec des petits pas, on est précis mais ça prend beaucoup de temps. Avec des grands pas, on va vite mais on risque de sauter par-dessus le point le plus bas, et si le pas est vraiment trop grand, on s'en éloigne au lieu de s'en approcher.

Exemple : je suis sur une colline, les yeux bandés, et je veux rejoindre la vallée. Je sens avec mes pieds de quel côté ça descend, je fais un pas dans ce sens, puis je recommence. Avec de petits pas, j'y arrive sûrement mais lentement. Avec des pas de géant, je peux passer au-dessus de la vallée sans m'arrêter.

## Différence entre apprentissage supervisé et non supervisé
En supervisé, l'IA s'entraîne avec les réponses. À chaque essai, elle compare ce qu'elle a prédit avec la bonne réponse et se corrige : c'est comme ça qu'elle apprend. Ensuite, on vérifie si elle est bonne sur des données de test qu'elle n'a jamais vues.

En non supervisé, l'IA n'a pas les réponses. Elle a quand même un objectif : obtenir le meilleur "score" possible, calculé à partir des données seules (par exemple, que les images d'un même groupe se ressemblent). Ce type de problème s'appelle le clustering.

Exemple : l'IA doit classer des photos.
- En supervisé, on lui donne trois catégories : chien, chat et oiseau. Elle range les photos dedans et regarde les réponses pour apprendre.
- En non supervisé, il n'y a aucune catégorie. L'IA fait des tas de photos qui se ressemblent, obtient un score, puis recommence pour l'améliorer. C'est ensuite à l'humain de regarder les tas et de comprendre ce qu'ils contiennent.

## Pourquoi on ne fait pas de descente de gradient sur les hyperparamètres ?
- Beaucoup d'hyperparamètres sont des choix à valeurs séparées, comme 3 ou 4 neurones, ou le degré d'un polynôme. On ne peut pas descendre une pente quand il n'y a pas de pente.
- Si on les réglait avec l'erreur d'entraînement, on choisirait toujours le modèle le plus compliqué, donc de l'overfitting. Il faut les choisir sur des données de validation.
- Chaque essai demande un entraînement complet, donc ce serait beaucoup trop long.

On teste donc plusieurs valeurs avec du grid search, du random search ou Optuna.