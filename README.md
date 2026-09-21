# Banc d'expériences pré-enregistrées en apprentissage auto-supervisé

Ce dépôt regroupe des expériences sur les objectifs et les architectures d'apprentissage
auto-supervisé (SSL). Elles testent huit thèses, chacune traduite en une prédiction précise et
en un critère de réfutation, écrits avant le premier run. Le dépôt contient le code, les
résultats bruts et le journal de tout ce qui a tourné, résultats négatifs et lectures retirées
compris.

L'échelle est petite : CIFAR-10 ou COCO en 128 px, un ResNet-18 ou de petits réseaux
convolutifs, un seul A100 sur Colab. Rien ici ne vise l'état de l'art. Les expériences mesurent
si un mécanisme se produit ou non, à budget identique entre les bras comparés.

## Protocole

- **Prédiction et falsifieur écrits avant le run**, dans le code (dictionnaires `PREREG_*` en
  tête des scripts). Le verdict est calculé par le script.
- **Trois graines au minimum.** En dessous, `aggregate.py` ramène le verdict à « insuffisant »,
  quoi que le script ait conclu.
- **Budgets identiques entre bras.** `matched_budget_check` compare les pas, le batch et les
  passes forward, et lève une erreur s'ils diffèrent.
- **Un run `--smoke` ne compte jamais.** Il vérifie que le pipeline tourne, pas qu'une thèse
  tient, et les scripts l'écrivent en clair à la fin.
- **Aucun seuil n'est modifié après lecture des données.** Dans l'entrée E-005, un régime sort à
  8,95 % pour un plancher fixé à 10 % : le verdict reste « régime vide ».
- **`JOURNAL.md` contient une entrée par run**, avec une rubrique « ce que ce run ne décide
  pas ». Une lecture erronée n'est pas effacée : la correction est écrite à la suite (entrées
  A-002 et A-005).

## État des expériences

| Exp. | Thèse | Prédiction | Ce qui la réfuterait | État |
|---|---|---|---|---|
| **A1** | T1 | `\|L(Z) − L(ZQ)\|` est à la précision machine pour la perte de Gram et InfoNCE, macroscopique pour une tête à prototypes, et revient à la précision machine si on tourne aussi les prototypes | un résidu qui ne décroît pas en float64 côté relationnel, ou un résidu à la précision machine côté prototypes non tournés | **confirmée** (A-001) |
| **A2** | T2 | avec la perte de Gram seule, la perte tend vers 0 pendant que le rang effectif des embeddings s'effondre ; le terme variance/covariance sur `z` non normalisé l'empêche | pas d'effondrement sans ce terme, ou effondrement avec | **en partie infirmée** (A-001 à A-005) |
| **B** | T1 | à batch contrôlé, la performance varie davantage à travers un balayage d'hyperparamètres côté prototypes que côté relationnel, et l'écart se creuse en queue longue | dispersion égale ou inverse, ou couplage au batch du même ordre que l'effet | pas lancée |
| **C1** | T3 | des têtes entraînables sous un pire cas (max ou LSE) convergent les unes vers les autres, des têtes gelées non | pas de convergence des têtes entraînables | lancée, pas encore journalisée |
| **C2-C3** | T3 | le pire cas adouci sur un panel de têtes gelées améliore la pire tâche aval, à moyenne égale ou un peu inférieure | pire tâche inchangée ou dégradée | pas lancées |
| **D1** | T5 | une politique d'augmentation entraînée à minimiser la perte SSL s'effondre vers l'identité ; sous budget adversarial, seul le budget déplace le résultat | la politique apprise déplace les résultats à budget gelé | pas lancée |
| **D2** | T6 | les courbes « augmentations prudentes » et « augmentations fortes » se croisent entre un domaine standard et un domaine où l'invariance supposée est fausse ; le fine-tuning récupère ce que la sonde linéaire ne récupère pas | pas de croisement, ou la sonde linéaire récupère tout | pas lancée |
| **E0** | T4 | la portée de corrélation du résidu d'un prédicteur masqué sépare les objets des textures mieux que le critère naïf « le résidu est structuré » | AUC de la portée à peu près égale à celle du critère naïf | **non mesurable à cette échelle** : trois régimes vides (E-001 à E-005) |
| **E1** | T4 | à FLOPs égaux, un routeur fondé sur ce critère bat l'absence de routeur et bat le critère naïf | le naïf fait aussi bien | bloquée par E0 |
| **F1** | T7 | le fine-tuning gagne en pic par tâche, l'encodeur gelé gagne en pire cas et en variance | le gelé perd aussi en pire cas, ou ses poids bougent | pas lancée |
| **F2** | T8 | un latent sans perte propre ni décodeur transfère mieux ; un bras stop-gradient sépare « le décodeur déforme » de « le décodeur consomme du budget » | pas d'écart, ou écart entièrement expliqué par le bras stop-gradient | pas lancée |

## Résultats

### A1 : invariance par rotation

Une cible relationnelle compare des matrices de Gram, `A Aᵀ`. Si la même rotation `Q` est
appliquée à tous les embeddings, `(AQ)(AQ)ᵀ = A Aᵀ` : la perte ne change pas. Une tête à
prototypes compare chaque embedding à des vecteurs appris, qui fixent un repère dans l'espace
latent.

En float64, le résidu de rotation vaut 4,0 × 10⁻¹⁹ pour la perte de Gram et 1,1 × 10⁻¹⁶ pour
InfoNCE. Il vaut 1,45 × 10⁻² avec des prototypes fixes tirés au hasard, et 5,74 avec des
prototypes ajustés au batch. Si les prototypes sont tournés aussi, il retombe à 3,0 × 10⁻¹⁷.
L'invariance est donc brisée par le repère appris, pas par le softmax.

C'est un fait algébrique vérifié numériquement. Ses conséquences pratiques éventuelles sont
l'objet de l'expérience B, qui n'a pas tourné.

### A2 : perte de Gram et terme variance/covariance

**Montage.** Deux vues augmentées de chaque image passent dans le même réseau : un ResNet-18,
puis un projecteur vers 128 dimensions. La perte d'accord aligne la matrice de Gram des
embeddings normalisés d'une vue sur celle de l'autre vue, avec stop-gradient sur la cible, dans
les deux sens. Il n'y a ni maître EMA ni prédicteur. Selon le bras s'ajoute le terme
variance/covariance de VICReg. Sont mesurés le kNN et une sonde linéaire sur la sortie du
backbone, et le rang effectif à plusieurs endroits. CIFAR-10, 6 000 pas, batch 256, trois
graines par bras.

| bras | ce qui change | kNN % | sonde linéaire % | rang des embeddings /128 | rang du backbone /512 |
|---|---|---|---|---|---|
| `gram` | perte d'accord seule | 35,7 | 37,3 | 9,0 | 312 |
| `gram_vicreg` | + terme sur `z` non normalisé, comme dans VICReg | 33,8 | 41,4 | 72,3 | 209 |
| `gram_vicreg_sg` | le gradient du terme est coupé avant le backbone | 33,1 | 38,5 | 7,3 | 144, instable |
| `gram_vicreg_sphere` | terme calculé sur `z` normalisé, cible de variance γ = 1 | 44,5 | 44,5 | 123,3 | 446 |
| `gram_vicreg_sphere_gamma` | γ = 1/√d | 41,1 | 42,1 | 123,3 | 448 |
| `gram_vicreg_sphere_gamma_half` | γ = 0,5/√d, bras ajouté après coup | 39,3 | 40,5 | 95,9 | 454 |

Le rang des embeddings est non centré, mesuré sur 512 vues augmentées. Le rang du backbone est
centré, mesuré sur les 10 000 images propres du test. Les deux premiers bras viennent du run
A-003, les quatre autres du run A-004, à configuration identique. Deux runs identiques diffèrent
d'environ un point, à cause du non-déterminisme du GPU.

**Prédictions.** La perte de Gram seule ne s'effondre pas complètement, contrairement à la
prédiction : le rang des embeddings reste bas mais le kNN tient, très au-dessus du hasard. Le
terme variance/covariance remonte le rang des embeddings, comme prévu. Deux hypothèses sur son
mécanisme sont infirmées : couper son gradient avant le backbone ne récupère pas le kNN, et ce
bras est instable d'une graine à l'autre ; une cible de variance atteignable remodèle le
backbone autant qu'une cible saturée.

**Observations hors prédiction.** Les trois points suivants ont été vus après coup. Ce sont des
observations, pas des tests.

1. Le terme calculé sur `z` normalisé, variante que le protocole considérait comme une erreur,
   est la meilleure sur toutes les mesures : 44,5 % de kNN contre 35,7 % sans le terme.
2. Le signe de l'effet dépend de la sonde. Ajouter le terme sur `z` non normalisé fait perdre
   1,9 point de kNN et gagner 4,1 points de sonde linéaire, sur les mêmes features et les mêmes
   graines, sans chevauchement entre graines.
3. Le rang ne donne pas la même lecture selon l'endroit où il est mesuré. Sur les trois bras
   « sphere », le kNN descend (44,5, puis 41,1, puis 39,3) pendant que le rang du backbone monte
   (446, puis 448, puis 454). Entre les deux bras extrêmes, le rang non centré des embeddings
   suit le kNN, 123 contre 96. Cet écart vient presque entièrement de la direction moyenne des
   embeddings : une fois centré, il reste 123,3 contre 122,7. Le bras intermédiaire a le même
   rang d'embeddings que le meilleur, à la décimale près, avec 3,4 points de kNN en moins.

**Lecture retirée.** La première lecture de A2 disait que le terme multipliait le rang par cinq
et coûtait deux points de kNN. Elle comparait le rang du projecteur au kNN du backbone, deux
espaces différents ; l'entrée A-002 la retire. L'entrée A-005 reprend la question de l'espace de
mesure : passer de 512 à 10 000 échantillons monte tous les rangs d'environ 23 % sans changer
les écarts entre bras de plus d'un point, alors que mesurer sur des vues augmentées plutôt que
sur des images propres divise ces écarts par deux.

### E0 : portée de corrélation du résidu

**Idée.** Un petit réseau convolutif prédit des blocs cachés d'une image. La thèse T4 propose de
regarder sur quelle distance ses erreurs restent corrélées, et de comparer cette distance à son
champ réceptif : une erreur corrélée plus loin que le champ réceptif signalerait un objet, une
erreur à courte portée une texture. Le test se fait sur COCO-Stuff en 128 px, avec les
étiquettes humaines « objet » ou « fond » par cellule de 8 px, et se lit comme une AUC.

**Trois designs, trois régimes vides.** À chaque fois, la précondition du test n'était pas
remplie : presque aucune cellule n'avait d'erreur corrélée au-delà du champ réceptif, et le
critère ne pouvait se déclencher sur rien. Les causes diffèrent.

- E0b : le prédicteur voyait trop peu (champ réceptif de 11 px, 75 % de l'image cachée). Il
  apprenait un flou local en 500 pas, puis plus rien.
- E0c : avec un champ réceptif de 23 px et 33 % caché, il apprend jusqu'au bout. Mais les trous
  de 8 px sont plus petits que ce qu'il voit : il interpole et n'a jamais à deviner. Son erreur
  devient un grain fin, le même sur un visage et sur de l'herbe.
- E0d : avec des trous de 32 px, il doit extrapoler, et l'erreur au centre du trou vaut 2,2 fois
  celle des coins. Mais la statistique lit alors la géométrie des masques : la profondeur dans
  le trou explique 8,92 % de sa variance, l'étiquette objet ou fond 0,0245 %.

Forcer le réseau à extrapoler demande des masques grands et réguliers, et de tels masques
impriment leur propre portée dans l'erreur. Dans les trois runs, ce qui sépare un peu les objets
des fonds est la taille de l'erreur (AUC de 0,55 à 0,62), pas sa portée. Ce n'est pas une
réfutation de T4 : le test n'a jamais pu se déclencher.

## Limites

- **Petite échelle.** CIFAR-10, un ResNet-18, 6 000 pas. Les modèles de A2 sont loin de la
  convergence : 44 % de kNN là où un entraînement complet dépasse 85 %. Un écart mesuré ici est
  un écart entre bras à budget identique, jamais un chiffre comparable à la littérature.
- **CIFAR-10 n'a que 10 classes.** Les rangs effectifs mesurés sont très au-dessus de ce nombre,
  dans la zone où le rang n'a plus de raison théorique de prédire la précision d'une
  classification.
- **Une seule pondération** du terme variance/covariance (25 et 1), un seul taux
  d'apprentissage, et un clip de gradient à 5,0 qui touche aussi les bras sains.
- **Le bras `gram_vicreg_sphere_gamma_half` a été ajouté après coup.** Ses chiffres décrivent,
  ils ne confirment rien.
- **Des artefacts sont perdus.** Les checkpoints et les CSV du run A-003 ont disparu avec une
  machine Colab recyclée avant la synchronisation : il n'en reste que la sortie console,
  `results/A2v2_long/RUN_LOG.txt`. Du premier run à 1 500 pas, il ne reste que
  `results/A/A2_espaces.json`. Le journal le signale aux endroits concernés.
- **La série E n'a tourné qu'avec une graine par design.** Ce sont des lectures pilotes, pas des
  verdicts.
- **B, C2-C3, D et F n'ont pas tourné.** Aucune conclusion ne porte sur elles.

## Reproduire

```bash
python lib/harness.py                        # auto-test du socle, 3 secondes
python xp/xp_A_diagnostics.py --smoke        # vérifie le pipeline, jamais une thèse
python xp/xp_A_diagnostics.py --exp a1       # A1, 20 secondes sur CPU
```

Le run A2 à 6 000 pas :

```bash
python xp/xp_A_diagnostics.py --exp a2 --clip 5.0 --steps 6000 \
    --arm gram --arm gram_vicreg --arm gram_vicreg_sg \
    --arm gram_vicreg_sphere --arm gram_vicreg_sphere_gamma --arm gram_vicreg_sphere_gamma_half
```

La série E sur COCO-Stuff. Les commandes de téléchargement des données
sont dans `python xp/xp_E0d_coco.py --help`.

```bash
python xp/xp_E0d_coco.py --seed 0 --data-root <dossier>/cocostuff
```

Tous les chiffres de A2 cités plus haut se recalculent sans GPU à partir des fichiers
`results/A2v2_repair/*.summary.json` et de `results/A2v2_long/RUN_LOG.txt`.

**Sur Colab**, le code reste sur Drive, les données et les résultats sur le disque de la
machine, avec une synchronisation des résultats vers Drive. Relire CIFAR à chaque époque à
travers le montage Drive fait attendre le GPU, et les checkpoints n'y sont pas écrits de façon
fiable. `colab_run.py` fait ce partage, restaure les résultats au démarrage pour que `--resume`
reprenne après une coupure, et resynchronise toutes les deux minutes et à la sortie.

```bash
python colab_run.py --drive /content/drive/MyDrive/<dossier> -- python xp/xp_C_panel.py --exp c1
```

**Une expérience entière tourne sur un seul GPU, du début à la fin.** L'arithmétique en
précision mixte n'est pas la même sur un A100 et sur un T4, et `matched_budget_check` ne le voit
pas. Deux bras entraînés sur deux GPU différents ne se comparent pas.

## Fichiers

```
lib/harness.py            socle commun : données, encodeurs, objectifs, sondes, budget, reprise
xp/xp_A_diagnostics.py    A1 et A2
xp/xp_B ... xp_F          les autres groupes, chacun avec sa prédiction et son falsifieur en tête
xp/xp_E0b, E0c, E0d       les trois designs successifs de E0 sur COCO-Stuff
xp/calib_ruler_E0c.py     calibrage de la mesure de portée sur des champs synthétiques
reanalyse_A*.py           relectures de A2 sans réentraînement
aggregate.py              balaie results/ et donne un verdict par thèse
JOURNAL.md                une entrée par run, les plus récentes en haut
results/                  CSV, résumés JSON et tables par région de ce qui a tourné
colab_run.py              lanceur Colab décrit plus haut
colab_setup.ipynb         notebook de mise en route sur Colab
```
