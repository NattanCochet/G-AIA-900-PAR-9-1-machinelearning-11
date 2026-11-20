# Task 2 : MNIST, temps pour atteindre 0,97 de test accuracy

## Résumé

État au 19 novembre 2026. Réseau : CNN de la baseline du cours, optimiseur Adam, batch de 32 images, learning rate 0,001, cible 0,97 de test accuracy.

| Étape | Matériel | Temps d'apprentissage | Gain |
|---|---|---|---|
| Référence : images converties par le DataLoader à chaque pas | Tesla T4 (Colab) | 9,3 s | |
| MNIST chargé une fois sur le GPU (`--data-on-device`) | Tesla T4 (Colab) | 4,2 s | ×2,2 |
| Même réglage sur un PC local | RTX 3080 | 4,2 s | aucun |
| Même réglage, sur le processeur seul | Ryzen 7 3800XT | 11,4 s | |

Moyennes sur 5 graines (Colab) ou 3 graines (PC local).

Ce qu'on sait à ce stade :
1. **Le premier goulot était le DataLoader** : la conversion des images sur le CPU prenait 68 % d'un pas. Charger MNIST une fois sur le GPU divise le temps d'apprentissage par 2,2 (sections 3 et 4).
2. **Le learning rate change le nombre de pas, pas le coût d'un pas.** Le meilleur compromis est entre 0,001 et 0,003 ; à 0,01, la cible n'est jamais atteinte (section 6).
3. **Une RTX 3080 ne va pas plus vite qu'un T4.** Le profiler montre pourquoi : avec des batchs de 32, le GPU ne calcule que 10 à 13 % du temps d'un pas et attend le CPU le reste du temps (section 7).
4. **Un pas de 512 images coûte autant qu'un pas de 32** : augmenter le batch size est la prochaine piste (section 8).

## 1. Protocole

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

### Matériel (Google Colab)
- GPU : Tesla T4
- CPU : Intel Xeon @ 2,00 GHz, 2 threads
- PyTorch : 2.14.1+cu130

## 2. Résultats par défaut (Colab, DataLoader)

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

### Ce qu'on peut en conclure

1. **Les mesures sont très reproductibles, et la variation vient du nombre de pas.** Le temps par pas est quasi identique (5,6 ms) d'une graine à l'autre. Si les temps vont de 8,6 à 11,4 s, c'est parce que certaines graines atteignent 0,97 plus tôt que d'autres, pas parce que la machine est bruitée. Le temps est donc à peu près **nombre de pas × 5,6 ms**.
2. **Le chargement est maintenant à 0,07 s.** Les 3,17 s de la première exécution étaient bien le téléchargement de MNIST.
3. **Ce qui ressemble au goulot : le chargement des images sur le CPU.**
   - Les évaluations et l'inférence prennent chacune environ 1 s pour 10 000 images, soit 0,1 ms par image, ce qui est lent pour un GPU et un si petit réseau.
   - Le Xeon n'a que 2 threads et chaque image passe par une conversion CPU à chaque accès (`ToTensor` dans le DataLoader).
   - Si ce coût est du même ordre à l'entraînement, il représenterait environ 3 ms des 5,6 ms d'un pas, soit plus de la moitié. C'est une estimation à partir de l'inférence, pas une mesure.
   - Cette hypothèse est confirmée par le test ci-dessous.

## 3. Test de confirmation du goulot

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

## 4. Correction : MNIST chargé sur le GPU (`--data-on-device`)

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
- **Étape suivante** : balayer le learning rate (fait en section 6) puis le batch size (section 8), maintenant que le chargement ne domine plus.

## 5. Passage sur un PC local (RTX 3080)

Les mesures des sections 6 et 7 sont faites sur un PC fixe, plus puissant que la machine Colab :

| | PC local | Colab (sections 1 à 4) |
|---|---|---|
| GPU | NVIDIA GeForce RTX 3080, 10 Go (architecture Ampere) | Tesla T4 |
| CPU | AMD Ryzen 7 3800XT, 8 cœurs / 16 threads, 3,9 GHz | Intel Xeon @ 2,00 GHz, 2 threads |
| RAM | 32 Go | — |
| Système | Windows 11 | Linux |
| PyTorch | 2.15.0+cu134 (CUDA 13.4, pilote NVIDIA 621.24) | 2.14.1+cu130 |

- Le `pyproject.toml` du dépôt installe PyTorch sans GPU. Sur ce PC, une configuration `uv` locale, non versionnée, installe à la place la version CUDA 13.4 : les commandes ci-dessous y sont lancées avec `uv run --extra cu134 python …` au lieu de `uv run python …`.
- Chaque run enregistre maintenant dans `results/task2_mnist.csv`, en plus des réglages et des temps : un `tag` (option `--tag`) pour regrouper les runs d'une même expérience, le commit git du code (`git_commit`, suffixé `-dirty` si le code a été modifié), le nom commercial du CPU et la version de CUDA.

## 6. Balayage du learning rate

Un seul paramètre change : le learning rate (0,0003 ; 0,001 ; 0,003 ; 0,01). Le reste est fixe : MNIST chargé une fois en mémoire (`--data-on-device`), batch de 32, Adam, contrôle tous les 100 pas, au plus 20 epochs. 3 graines (0, 1, 2) par valeur.

```powershell
foreach ($lr in 0.0003, 0.001, 0.003, 0.01) { foreach ($seed in 0, 1, 2) { uv run python -m task2_mnist.time_to_target --data-on-device --lr $lr --seed $seed } }
```

### Sur le CPU (Ryzen 7 3800XT, 16 threads)

| Learning rate | Cible atteinte | Pas pour atteindre 0,97 | Temps d'apprentissage | Temps par pas | Inférence (10 000 images) |
|---|---|---|---|---|---|
| 0,0003 | 3 / 3 | 3 867 (± 351) | 22,7 s (± 2,7) | 5,9 ms | 0,40 s |
| 0,001 | 3 / 3 | 1 867 (± 208) | 11,4 s (± 0,9) | 6,1 ms | 0,43 s |
| **0,003** | 3 / 3 | **1 467** (± 58) | **8,8 s** (± 0,5) | 5,9 ms | 0,41 s |
| 0,01 | **0 / 1** | non atteint : 0,944 après 20 epochs (37 500 pas) | 199,6 s | 5,3 ms | 0,40 s |

### Sur le GPU (RTX 3080)

| Learning rate | Cible atteinte | Pas pour atteindre 0,97 | Temps d'apprentissage | Temps par pas | Inférence (10 000 images) |
|---|---|---|---|---|---|
| 0,0003 | 3 / 3 | 4 000 (± 100) | 7,6 s (± 0,1) | 1,8 ms | 12 ms |
| **0,001** | 3 / 3 | 1 967 (± 208) | **4,2 s** (± 0,2) | 2,0 ms | 7 ms |
| **0,003** | 3 / 3 | **1 533** (± 208) | **4,3 s** (± 0,4) | 2,6 ms | 7 ms |
| 0,01 | **0 / 3** | non atteint : 0,914 à 0,946 après 20 epochs | 65,6 s (± 0,9) | 1,7 ms | 12 ms |

± : écart-type sur les 3 graines. Temps d'apprentissage : chargement + setup + entraînement, sans les évaluations de contrôle. Temps par pas : temps d'entraînement ÷ nombre de pas.

### Ce qu'on peut en conclure

1. **Le learning rate change le nombre de pas, pas le coût d'un pas.** Sur CPU, un pas coûte environ 6 ms quel que soit le learning rate. Le temps d'apprentissage suit donc le nombre de pas : environ 3 900 pas à 0,0003 contre 1 500 à 0,003.
2. **Trop grand, il empêche d'atteindre la cible.** À 0,01, aucun run n'atteint 0,97 en 20 epochs, sur CPU comme sur GPU : la test accuracy reste entre 0,91 et 0,95. Les pas d'Adam sont trop grands, l'optimisation oscille au lieu de converger.
3. **Le meilleur compromis est entre 0,001 et 0,003** : 4,2 à 4,3 s sur le GPU, 8,8 s sur le CPU. Sur le GPU, l'écart entre ces deux valeurs n'est pas significatif avec 3 graines.
4. **Le GPU n'apporte qu'un facteur 2 à 3 par rapport au CPU** (×2,7 à lr 0,001). C'est peu pour une RTX 3080 : la section 7 explique pourquoi.
5. **La même graine ne donne pas le même nombre de pas sur CPU et sur GPU** (lr 0,0003, graine 0 : 3 500 pas sur CPU, 4 100 sur GPU). Le tirage des batchs (`torch.randperm`) ne produit pas la même suite de nombres sur les deux appareils.
6. **La RTX 3080 n'est pas plus rapide que le T4** : à lr 0,001 avec les données sur le GPU, 4,2 s et 2,0 ms par pas, contre 4,2 s et 1,89 ms sur Colab (section 4). Sa puissance de calcul n'est donc pas ce qui limite.

**Réserves :**
- Pendant les runs GPU, l'environnement Python était abîmé : `uv` réinstallait PyTorch (environ 10 s de copie de fichiers) juste avant chaque run, ce qui a pu les perturber. C'est probablement la cause des 2,6 ms par pas à lr 0,003, alors que le learning rate ne devrait pas changer le coût d'un pas (1,7 à 2,0 ms pour les autres valeurs). Ce balayage GPU est à refaire maintenant que l'environnement est réparé.
- Sur CPU, lr 0,01 n'a qu'une graine : la série a été arrêtée avant les graines 1 et 2.

## 7. Profilage avec torch.profiler

### Protocole

`--profile` remplace le run « jusqu'à 0,97 » et n'écrit rien dans le CSV :
1. 1 s d'entraînement pour que le GPU monte en fréquence, puis 500 pas chronométrés **sans** profiler : c'est le vrai coût d'un pas ;
2. une trace `torch.profiler` (côté CPU et côté GPU) de 20 pas d'entraînement, après 5 pas de chauffe non gardés ;
3. la même chose pour l'inférence sur les 10 000 images de test (5 passages chronométrés, 1 enregistré) ;
4. le **taux d'occupation du GPU** : temps de calcul GPU d'un pas (mesuré par le profiler) ÷ temps d'un pas (mesuré sans profiler).

Le profiler ralentit le côté CPU : un pas dure 5,6 ms sous profiler contre 2,25 ms sans. Ses durées CPU servent à comprendre, jamais à mesurer. Les durées des calculs GPU, elles, ne sont pas modifiées.

```powershell
uv run python -m task2_mnist.time_to_target --data-on-device --batch-size 32 --profile
uv run python -m task2_mnist.time_to_target --data-on-device --batch-size 512 --profile
uv run python -m task2_mnist.plot_profile
```

Les traces sont écrites dans `task2_mnist/traces/` (non versionnées) et s'ouvrent sur https://ui.perfetto.dev.

### Résultats (RTX 3080, données sur le GPU)

| | Batch 32 | Batch 512 |
|---|---|---|
| Temps d'un pas, sans profiler | 2,15 à 2,96 ms (5 lancements) | 1,81 à 2,08 ms (4 lancements) |
| Temps de calcul GPU par pas | 0,28 à 0,30 ms | 1,19 ms |
| **Occupation du GPU pendant un pas** | **10 à 13 %** | **57 à 66 %** |
| Calculs GPU (kernels) lancés par pas | 37 | 33 |
| Images traitées par pas | 32 | 512 |

Inférence (10 000 images par paquets de 2 048, même modèle) : 7,1 à 7,7 ms par passage, dont 6,5 à 6,9 ms de calcul GPU, soit **84 à 95 % d'occupation**.

### Ce que montre la trace

![Frise CPU et GPU de 3 pas d'entraînement, avec un batch de 32 et un batch de 512](../../results/task2_mnist_profile_timeline.png)

La figure (`task2_mnist/plot_profile.py`) reprend 3 pas des traces. En bleu, les moments où le CPU exécute une opération PyTorch ; en orange, ceux où le GPU calcule. Les pourcentages du titre sont mesurés sous profiler, qui étire le côté CPU : ils sont plus bas que les vrais taux d'occupation du tableau.

- **Batch 32 : le GPU attend presque tout le temps.** Chaque pas lance 37 petits calculs l'un après l'autre (forward, backward, Adam). Chacun dure quelques microsecondes (un produit de matrices `aten::mm` : 6 µs ; le backward de la convolution : 42 µs), moins que le temps que met Python à préparer et lancer le suivant. Les traits orange sont isolés au milieu de grands vides.
- **Batch 512 : à peu près autant de calculs, mais 16 fois plus gros.** Le temps d'un pas ne change presque pas (environ 2 ms), parce qu'il est fixé par le CPU qui lance les opérations. Le GPU, lui, est enfin occupé une bonne partie du temps.
- **Le CPU travaille en continu.** Les 27 % de vide sur la ligne bleue correspondent au code Python exécuté entre deux opérations PyTorch, que le profiler n'enregistre pas ici.

### Détails tirés du tableau du profiler (batch 32)

- **Adam** occupe environ 0,5 ms de chaque pas sur la ligne du GPU (`Optimizer.step#Adam.step`), alors que ses calculs (`_foreach_addcdiv_`, `_foreach_lerp_`, `_foreach_sqrt`…) ne totalisent qu'environ 80 µs : le reste est de l'attente entre ces petites opérations. `torch.optim.Adam(..., fused=True)` les regrouperait en une seule.
- **Les convolutions tournent déjà en TF32** (`tf32` dans le nom des kernels cuDNN) : cuDNN l'active par défaut sur les GPU Ampere. Les produits de matrices des couches linéaires (`ampere_sgemm_…`), eux, sont en float32.

### Conclusions

1. **Avec des batchs de 32, l'entraînement est limité par le CPU, pas par le GPU** : le GPU calcule 10 à 13 % du temps. C'est pour cela qu'une RTX 3080 ne fait pas mieux qu'un T4 : sur les deux machines, c'est le lancement des opérations depuis Python qui fixe la durée d'un pas.
2. **Un pas de 512 images coûte à peu près autant qu'un pas de 32** : augmenter le batch size devrait réduire fortement le nombre de pas sans allonger chaque pas.
3. **L'inférence, elle, est limitée par le GPU** (84 à 95 % d'occupation) : pour l'accélérer, il faut réduire le calcul lui-même (précision réduite, modèle plus petit), pas le nombre d'opérations.
4. **Le temps d'un pas varie d'un lancement à l'autre** (2,15 à 2,96 ms pour le batch 32) : comme le CPU est le facteur limitant, toute application ouverte en arrière-plan ralentit l'entraînement. Il faut fermer les applications lourdes et répéter les mesures.

## 8. Prochaines étapes

Par ordre de gain attendu. Un seul changement à la fois, 3 graines et un `--tag` par série.

1. **Balayer le batch size** (32 à 1 024) à learning rate fixe, avec un contrôle tous les 10 pas (`--eval-every 10`). Avec de gros batchs, la cible est atteinte en quelques centaines de pas : un contrôle tous les 100 pas pourrait faire continuer l'entraînement jusqu'à 99 pas de trop.
   ```powershell
   foreach ($bs in 32, 64, 128, 256, 512, 1024) { foreach ($seed in 0, 1, 2) { uv run python -m task2_mnist.time_to_target --data-on-device --lr 0.003 --batch-size $bs --eval-every 10 --seed $seed --tag bs_sweep_gpu } }
   ```
2. **Faire grandir le learning rate avec le batch size**, selon la règle de Goyal et al. (*Accurate, Large Minibatch SGD*, 2017) : c'est la méthode tirée d'un article qu'exige le sujet.
3. **Réduire le nombre d'opérations par pas** : `Adam(fused=True)`, puis `torch.compile` si Triton fonctionne sous Windows (sinon sur Colab).
4. **Réduire la précision** : float16 avec `torch.autocast` (Micikevicius et al., *Mixed Precision Training*, 2018). Effet attendu surtout sur l'inférence et sur les gros batchs.
5. **Tracer les graphiques** demandés par le sujet : temps pour atteindre 0,97 en fonction du learning rate, puis du batch size.
6. **Refaire le balayage du learning rate sur GPU** dans un environnement propre (voir les réserves de la section 6).
