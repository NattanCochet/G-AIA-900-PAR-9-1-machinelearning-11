# Task 2 : MNIST, temps pour atteindre 0,97 de test accuracy

## Protocole
- Script : `task2_mnist/time_to_target.py`, réglages par défaut
- Réseau : CNN de la baseline du cours (port PyTorch), optimiseur Adam, learning rate 1e-3, batch size 32
- Cible : **0,97 de test accuracy**, vérifiée tous les 100 pas
- Temps d'apprentissage = chargement + setup + entraînement, **sans** les évaluations de contrôle
- 3 graines (0, 1, 2), dans la même session Colab

```python
!python -m task2_mnist.time_to_target --seed 0
!python -m task2_mnist.time_to_target --seed 1
!python -m task2_mnist.time_to_target --seed 2
```

## Matériel (Google Colab)
- GPU : Tesla T4
- CPU : Intel Xeon @ 2,00 GHz, 2 threads
- PyTorch : 2.14.1+cu130

## Résultats

| Graine | Pas pour atteindre 0,97 | Test accuracy | Temps d'apprentissage | Chargement | Setup | Entraînement | Évaluations (exclues) | Inférence | Temps par pas |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1500 | 0,9705 | 8,59 s | 0,07 s | 0,18 s | 8,35 s | 15,23 s | 0,92 s | 5,57 ms |
| 1 | 2000 | 0,9704 | 11,40 s | 0,07 s | 0,18 s | 11,15 s | 21,38 s | 0,98 s | 5,58 ms |
| 2 | 1700 | 0,9701 | 9,77 s | 0,07 s | 0,18 s | 9,52 s | 17,28 s | 1,14 s | 5,60 ms |
| **Moyenne** | **1733** | | **9,9 s** (écart-type ≈ 1,4 s) | | | | | | **≈ 5,6 ms** |

### Sortie brute

```
hardware: {'device': 'cuda', 'cpu': 'Intel(R) Xeon(R) CPU @ 2.00GHz', 'cpu_threads': 2, 'torch': '2.14.1+cu130', 'gpu': 'Tesla T4'}
target 0.97 reached after 1500 steps (test accuracy 0.9705)
learning time (load + setup + train): 8.59 s
  load: 0.07 s
  setup: 0.18 s
  train: 8.35 s
  eval: 15.23 s
  inference: 0.92 s
hardware: {'device': 'cuda', 'cpu': 'Intel(R) Xeon(R) CPU @ 2.00GHz', 'cpu_threads': 2, 'torch': '2.14.1+cu130', 'gpu': 'Tesla T4'}
target 0.97 reached after 2000 steps (test accuracy 0.9704)
learning time (load + setup + train): 11.40 s
  load: 0.07 s
  setup: 0.18 s
  train: 11.15 s
  eval: 21.38 s
  inference: 0.98 s
hardware: {'device': 'cuda', 'cpu': 'Intel(R) Xeon(R) CPU @ 2.00GHz', 'cpu_threads': 2, 'torch': '2.14.1+cu130', 'gpu': 'Tesla T4'}
target 0.97 reached after 1700 steps (test accuracy 0.9701)
learning time (load + setup + train): 9.77 s
  load: 0.07 s
  setup: 0.18 s
  train: 9.52 s
  eval: 17.28 s
  inference: 1.14 s
```

## Ce qu'on peut en conclure

1. **Les mesures sont très reproductibles, et la variation vient du nombre de pas.** Le temps par pas est quasi identique (5,6 ms) d'une graine à l'autre. Si les temps vont de 8,6 à 11,4 s, c'est parce que certaines graines atteignent 0,97 plus tôt que d'autres, pas parce que la machine est bruitée. Le temps est donc à peu près **nombre de pas × 5,6 ms**.
2. **Le chargement est maintenant à 0,07 s.** Les 3,17 s de la première exécution étaient bien le téléchargement de MNIST.
3. **Ce qui ressemble au goulot : le chargement des images sur le CPU.**
   - Les évaluations et l'inférence prennent chacune environ 1 s pour 10 000 images, soit 0,1 ms par image, ce qui est lent pour un GPU et un si petit réseau.
   - Le Xeon n'a que 2 threads et chaque image passe par une conversion CPU à chaque accès (`ToTensor` dans le DataLoader).
   - Si ce coût est du même ordre à l'entraînement, il représenterait environ 3 ms des 5,6 ms d'un pas, soit plus de la moitié. C'est une estimation à partir de l'inférence, pas une mesure.
   - Cette hypothèse est confirmée par le test ci-dessous.

## Test de confirmation du goulot

Trois mesures séparées sur 300 pas (batch 32, même session T4) :
- **A** : chargement seul (le DataLoader est parcouru sans calcul)
- **B** : chargement + copie vers le GPU
- **C** : calcul seul (toujours le même batch, déjà sur le GPU)

| Étape | Temps par pas | Part du pas mesuré (5,6 ms) |
|---|---|---|
| Chargement (A) | 3,09 ms | ≈ 55 % |
| Copie CPU vers GPU (B − A) | 0,69 ms | ≈ 12 % |
| Calcul : aller, retour, Adam (C) | 1,58 ms | ≈ 28 % |
| Non expliqué | ≈ 0,24 ms | ≈ 4 % |
| **Total mesuré** | **5,6 ms** | 100 % |

- **Hypothèse confirmée** : chargement + copie font environ **68 %** d'un pas, le calcul seulement 28 %.
- Les étapes s'additionnent presque exactement (3,78 + 1,58 = 5,36 ms pour 5,6 mesurés) : le CPU et le GPU ne travaillent pas en parallèle.
- Le chargement coûte 0,097 ms par image, comme à l'inférence : c'est la même cause.
- **Prévision, à vérifier** : avec MNIST chargé une fois sur le GPU, un pas ne coûterait plus que le calcul (environ 1,6 à 1,8 ms), soit un gain proche de ×3.
- **Prochaine étape** : implémenter les données sur GPU et mesurer sur 3 à 5 graines dans la même session.