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
- **Prévision** : avec MNIST chargé une fois sur le GPU, un pas ne coûterait plus que le calcul (environ 1,6 à 1,8 ms), soit un gain proche de ×3. Mesuré ci-dessous : ×2,2 sur le temps d'apprentissage.

## Correction : MNIST chargé sur le GPU (`--data-on-device`)

Les images sont converties une fois et gardées en tenseurs sur le GPU, et les batchs sont tirés directement dessus (même prétraitement que `ToTensor`, vérifié bit pour bit). 5 graines (0 à 4) par mode, lancées à tour de rôle dans la même session Colab (Tesla T4), réglages par défaut.

| | DataLoader | Données sur GPU | Gain |
|---|---|---|---|
| **Temps d'apprentissage** (chargement + setup + entraînement) | 9,3 s (± 0,7) | 4,2 s (± 0,6) | **×2,2** |
| Entraînement seul | 9,0 s | 3,9 s | ×2,3 |
| Temps par pas | 5,15 ms | 1,89 ms | ×2,7 |
| Pas pour atteindre 0,97 | 1760 (± 182) | 2060 (± 313) | +17 % |
| Chargement | 0,07 s | 0,29 s | |
| Inférence (10 000 images) | 1,0 s | 0,020 s | ×50 |
| Évaluations de contrôle (total par run, exclues) | 16,9 s | 0,55 s | ×31 |
| Test accuracy à l'arrêt | 0,9701 à 0,9719 | 0,9700 à 0,9718 | identique |

- **Le goulot est levé** : temps d'apprentissage divisé par 2,2, sans perte de précision (les 10 runs atteignent 0,97).
- **Un peu sous la prévision (×3)** : un pas coûte 1,89 ms au lieu de 1,6 à 1,8 ms (le tirage des batchs sur le GPU a un petit coût, non mesuré), et les runs sur GPU demandent 17 % de pas en plus. Cet écart de pas est probablement du bruit : il n'est pas significatif avec 5 graines (t de Welch ≈ 1,85).
- Le chargement passe de 0,07 à 0,29 s (conversion + transfert), compté dans le temps d'apprentissage.
- L'inférence et les évaluations de contrôle sont 50 et 31 fois plus rapides : c'est la même cause, le chargement des images.
- Les résultats sont reproductibles : les graines 0, 1 et 2 s'arrêtent aux mêmes pas que dans la session précédente.
- **Prochaine étape** : balayer le learning rate puis le batch size, maintenant que le chargement ne domine plus.