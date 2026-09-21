# Journal de bord

Je le remplis **avant** de lancer un run, pas après. Trois semaines et deux cents CSV plus tard,
c'est la seule chose qui me dira pourquoi j'ai lancé ce run-là avec ce budget-là.

La colonne qui compte le plus est la dernière : **ce que ce run ne décide pas**. C'est celle
qu'on saute, et c'est celle qui empêche de surpromettre.

---

## Gabarit : copier-coller pour chaque run

```
### [ID]  AAAA-MM-JJ  :  <exp> / <bras>

Commande exacte     :
Graines             :                    (< 3 → aucune conclusion, seulement un signal)
Budget              : steps / bs / passes forward           budget apparié aux bras : oui / non
Wall-clock          :        h        GPU : A100 40GB / autre :
Reprise             : run unique / repris depuis step ____ / CSV vérifié sans trou : oui / non
Commit ou date des scripts :

Prédiction testée (une phrase, écrite AVANT le run) :

Ce qui la falsifierait (écrit AVANT le run) :

Résultat brut (le chiffre, avec son écart-type inter-graines) :

Verdict               : confirmée / infirmée / non concluante / insuffisant
Ce que ce run NE décide PAS :

Envoyé à quelqu'un ?  : non / oui → à qui, quel fichier d'email, quelle date
```

---

## Règles que le gabarit sert à faire respecter

1. **La prédiction et son falsifieur s'écrivent avant le run.** Écrits après, ils s'adaptent au
   résultat sans qu'on s'en aperçoive. C'est le mécanisme le plus courant d'auto-tromperie dans
   ce genre de travail, et il est indolore.
2. **Moins de 3 graines → jamais « confirmée ».** `aggregate.py` le force de toute façon, en
   downgradant le verdict à « insuffisant » quel que soit ce que le script a conclu.
3. **Un run `--smoke` n'entre jamais dans ce journal.** Les scripts impriment un avertissement
   en clair à la fin de chaque run smoke ; `aggregate.py` les exclut automatiquement. Si je suis
   tenté de noter un chiffre de smoke ici, c'est le signe qu'il faut lancer le vrai run.
4. **« Budget apparié : non » invalide la comparaison**, pas seulement l'affaiblit. Deux bras à
   batch ou à nombre d'étapes différents ne se comparent pas, et `matched_budget_check` lève
   plutôt que de laisser passer.
5. **Après une reprise, vérifier le CSV.** Colab coupe ; le `Run` du socle tronque les lignes
   en avance sur le checkpoint, mais vérifie quand même qu'il n'y a ni trou ni doublon de step
   avant d'agréger.
6. **La colonne « envoyé à quelqu'un »** existe pour une raison précise : si j'ai envoyé un
   chiffre et qu'il change ensuite, je dois pouvoir retrouver à qui, et le corriger moi-même.
   Un chiffre corrigé spontanément vaut plus qu'un chiffre juste du premier coup.

---

## Runs

<!-- Nouveaux runs à partir d'ici. Le plus récent en haut. -->

### [A-006]  2026-09-21  :  A2 / pré-enregistrement A2V3, pression de variance à gamma fixe et kNN standardisé (run non lancé)

```
Commande exacte     : python xp/xp_A_diagnostics.py --exp a2 --clip 5.0 --steps 6000
                      --arm gram --arm gram_vicreg --arm gram_vicreg_sphere
                      --arm gram_vicreg_sphere_dose51 --arm gram_vicreg_sphere_dose11
                      --arm gram_vicreg_sphere_interm51 --arm gram_vicreg_sphere_interm11
                      --arm gram_vicreg_sphere_early250
                      (lancé en trois groupes avec synchronisation entre chaque, puis une
                      dernière invocation sur les huit bras, qui ne réentraîne rien et calcule
                      les verdicts)
Graines             : 3 (0, 1, 2)
Budget              : 6 000 pas / bs 256 / 2 passes encodeur, identique pour les huit bras
Wall-clock prévu    : environ 5 h, 24 runs de 13 min             GPU : A100 (Colab)
Reprise             : à remplir après le run
Commit ou date des scripts : xp_A_diagnostics.py du 2026-09-21 (PREREG_A2V3), smoke passé
```

**Pourquoi ce run.** Dans A-004, trois bras « sphere » ne diffèrent que par la cible de variance
gamma. La charnière finit active sur 100 %, 51 % et 11 % des coordonnées, et le kNN fait 44,5,
41,1 et 39,3. Une cible plus basse est atteinte plus tôt : « gamma » et « quantité de pression de
variance reçue » sont un seul et même bouton. A-004 le notait comme une lecture, pas comme un
test. Ce run fait varier la pression à gamma fixe.

**Bras.** Tous à gamma = 1, contrainte jamais satisfiable sur la sphère. Seul le terme de
variance est modulé, le terme de covariance reste entier comme dans les bras d'origine.
- `dose51`, `dose11` : poids du terme de variance multiplié par 0,51 ou 0,11 à chaque pas.
- `interm51`, `interm11` : terme appliqué sur 51 % ou 11 % des pas (tirage fixé par la graine),
  absent sur les autres. Même pression intégrée que les bras « dose », persistance différente.
- `early250` : terme actif les 250 premiers pas, puis coupé.
- `gram_vicreg_sphere` est relancé dans le même run comme référence : deux runs identiques
  diffèrent d'environ un point de kNN (A-004), donc aucun seuil ne compare un bras nouveau à un
  chiffre d'un ancien run.

**Q3, prédiction (écrite AVANT le run).** La baisse de transfert entre les bras d'origine est
portée par la quantité de pression de variance, pas par la valeur de gamma : à gamma = 1,
réduire le terme à 11 % reproduit la baisse de kNN, et 51 % tombe entre les deux.

**Q3, ce qui la confirmerait :** moyenne sur les deux bras à 11 % de [kNN(sphere) − kNN(bras)]
≥ 3,0 points, ET kNN(sphere) > kNN(51 %) > kNN(11 %) dans les deux familles.
**Q3, ce qui la falsifierait :** les deux bras à 11 % finissent à moins de 1,5 point de
kNN(sphere). À gamma fixe la pression ne reproduit pas la baisse : c'est gamma qui comptait.
Ancres : la baisse d'origine est de 5,2 points ; 3,0 en vaut environ 60 % ; 1,5 couvre le bruit
entre runs plus une erreur standard d'une différence à 3 graines.

**Q4, prédiction (écrite AVANT le run).** Le désaccord de signe entre kNN et sonde linéaire de
A-003 (−1,94 point de kNN, +4,11 de sonde linéaire, gram contre gram_vicreg) est un artefact de
normalisation. La sonde linéaire standardise les features, le kNN ne fait que normaliser leur
longueur. Marks et al. (arXiv 2407.12210) rapportent que les deux sondes s'accordent (r = 0,99)
une fois les features normalisées, avec un z-score avant le kNN. Avec des features standardisées
sur les statistiques du train avant le kNN cosinus, kNN(gram_vicreg) − kNN(gram) devient positif.

**Q4, confirmée si** la différence de kNN standardisé dépasse +0,5 point. **Falsifiée si** elle
reste sous −1,0 point : le désaccord survit à la normalisation. **Prémisse :** la différence de
kNN brut doit se reproduire sous −1,0 point dans ce run, sinon « non concluante ».

**Exploratoire, jamais gaté :** dose contre intermittent à pression égale ; le bras early250 ; la
corrélation de Spearman, sur les bras de type sphere, entre le kNN et (i) `var_pressure_mean`,
part de la pression de variance reçue, lue sur la trajectoire sans étiquettes, (ii) le rang du
backbone, (iii) le rang non centré des embeddings sur images propres.

**Nouveau dans l'instrument.** Chaque run écrit `<nom>.weights.pt` (poids du modèle). Les
artefacts de A-003 et de `results/A` ont été perdus faute de cela. Chaque résumé porte aussi le
kNN standardisé et le rang des embeddings sur le test propre, centré et non centré, à N = 512
et N = 10 000.

**Déviation déclarée AVANT le run (2026-09-21).** Le run ne tournera pas sur A100 (Colab) mais
sur Kaggle, GPU T4 x2. Seuils, bras et prédictions sont inchangés. Ce qui change :
- `--no-amp` : le T4 n'a pas de bfloat16 natif, l'entraînement se fait en précision 32 bits.
  A-004 était en bf16. Les chiffres absolus de ce run ne sont donc pas directement comparables
  à ceux de A-004. Q3 et Q4 ne comparent que des bras de ce run, tous dans la même condition.
  `amp` est écrit dans la config de chaque résumé : un run fait dans une autre condition se
  verrait.
- Une invocation par graine (`--seed k --n-seeds 1`) au lieu d'une invocation à trois graines,
  avec deux files de quatre bras en parallèle, une par GPU, et `--workers 2`. Vérifié dans le
  code : `--seed` ne sert qu'à dériver la graine de chaque run (le sous-échantillonnage longue
  traîne, qui le lit aussi, est désactivé), donc la graine k lancée seule est le même run que la
  graine k d'une invocation à trois graines. Le nombre de workers change le flux
  d'augmentations, pas le budget, et il est le même pour les 24 runs.
- Commande par file : `python xp/xp_A_diagnostics.py --exp a2 --clip 5.0 --steps 6000 --no-amp
  --workers 2 --seed k --n-seeds 1` suivi des quatre `--arm` de la file.
- Les verdicts sont calculés après coup par `verdicts_a2`, la même fonction, sur les 24 résumés
  réunis, sans réentraînement.
- Wall-clock prévu : environ 1 h par run, 4 à 5 h par graine.

**Résultat brut :** [à remplir après le run]

**Verdict :** [à remplir après le run]

**Ce que ce run NE décide PAS (écrit AVANT) :**
- Une courbe dose-réponse le long du bouton qui règle la pression n'est pas un prédicteur sans
  étiquettes. Il faudrait des bras où la pression bouge indirectement (taux d'apprentissage,
  taille de batch, poids de covariance). Si Q3 est confirmée, c'est l'étape suivante, pas une
  conclusion.
- Une famille de perte, CIFAR-10, 6 000 pas, modèles loin de la convergence.
- Si Q4 est confirmée, l'observation « le signe dépend de la sonde » de A-003 est expliquée par
  un fait connu, et doit être présentée comme telle partout où elle apparaît.

**Envoyé à quelqu'un ?** non.

---

### [A-005]  2026-09-18  :  A2 / ce que N, l'augmentation et le split font au rang, aucun GPU

```
Commande exacte     : aucune. Relecture des 12 summaries de results/A2v2_repair (6 000 pas,
                      4 bras x 3 graines ; sphere seed2 lu dans A_summary.json, cf. A-004) et
                      de results/A/A2_espaces.json (1 500 pas, sortie de reanalyse_A2_espaces.py
                      --checkpoints du 2026-08-26, retrouvée dans Downloads le 2026-09-18).
Graines             : 3 par bras
Budget              : inchangé                              budget apparié : oui
Wall-clock          : 0 h            GPU : aucun
Reprise             : sans objet
```

**Statut : relecture post-hoc, exploratoire, pas un test.** Motivation : avant de citer où que
ce soit le contraste de rang de A-002 (rang du backbone, gram contre gram_vicreg à 1 500 pas :
0,04 σ sur vues augmentées à N = 512, −13,2 σ sur le test propre à N = 10 000), vérifier ce
qu'il mesurait. Les checkpoints de `results/A` sont perdus (ancien Drive devenu
inaccessible), donc la grille ajoutée à `reanalyse_A2_espaces.py` le 2026-09-17 ne peut pas
tourner sur ces modèles. Les runs à 6 000 pas, eux, loggent déjà quatre mesures du même rang.

**Correction d'une erreur de mes notes.** Ce contraste confondait trois axes : augmenté contre
propre, **pool d'entraînement contre split de test**, et N. Le centrage, lui, était apparié :
les deux chiffres sont non centrés (242,7 → 156,6 = −35,5 %, −13,2 σ ; A2_espaces.json). Mes
notes du 2026-09-17 disaient « centré contre non centré » à la place du split. Corrigé le
2026-09-18.

**Lecture attendue avant de regarder (notée le 2026-09-17, hors journal) :**
que N explique l'écart, c'est-à-dire que A-002 mesurait la taille d'échantillon et non
l'espace. **C'est l'inverse.**

**Résultat brut, rang effectif centré du backbone, 6 000 pas, moyenne sur 3 graines :**

| bras | aug, train, N = 512 | propre, train, N = 512 | propre, test, N = 512 | propre, test, N = 10 000 |
|---|---|---|---|---|
| sg (bimodal, exclu des moyennes) | 12,7 | 131,7 | 98,0 | 144,2 |
| sphere | 358,9 | 361,4 | 360,9 | 445,7 |
| sphere_gamma | 361,5 | 363,3 | 363,5 | 448,2 |
| sphere_gamma_half | 362,8 | 368,7 | 368,2 | 454,0 |

Effet de chaque axe sur le **niveau** du rang, 9 runs stables :

| axe | effet moyen | écart-type |
|---|---|---|
| augmentation (augmenté → propre, train, N = 512) | +3,4 (+0,9 %) | 2,0 |
| split (train → test, propre, N = 512) | −0,3 (−0,1 %) | 0,5 |
| N (512 → 10 000, propre, test) | **+85,1 (+23,4 %)** | 0,6 |

Effet de chaque case sur le **contraste entre bras** (différence, en σ de la différence) :

| contraste | aug, train, 512 | propre, train, 512 | propre, test, 512 | propre, test, 10 000 |
|---|---|---|---|---|
| sphere → half | +3,9 (2,8 σ) | +7,3 (8,2 σ) | +7,3 (10,2 σ) | +8,2 (10,1 σ) |
| sphere → gamma | +2,6 (1,9 σ) | +1,9 (1,5 σ) | +2,6 (2,2 σ) | +2,4 (1,9 σ) |
| gamma → half | +1,3 (0,8 σ) | +5,4 (3,5 σ) | +4,7 (3,3 σ) | +5,8 (3,8 σ) |

**Lecture.**
1. **N déplace beaucoup le niveau, peu les écarts.** Par construction, c'est une comparaison
   à un seul axe : le script calcule les features des 10 000 images de test une fois, puis
   prend les 512 premières lignes (`Hte[:512]`, xp_A_diagnostics.py) ; mêmes modèles, mêmes
   images, même centrage. Chacun des 9 runs monte de 23,2 à 23,6 %. L'écart sphere → half,
   graine par graine : +6,4 / +6,8 / +8,7 à N = 512, +7,4 / +7,5 / +9,9 à N = 10 000. Même
   signe, même ordre des graines, environ un point de plus, et séparation complète des deux
   bras dans les deux cas (plus petit half moins plus grand sphere : +6,4 puis +7,0). Les
   « 10 σ » du tableau reposent sur des écarts-types estimés sur 3 graines, donc très
   incertains, et les deux ne sont pas indépendants (mêmes modèles, images emboîtées) : c'est
   la séparation complète qui est solide, pas le chiffre en σ. Une première version de cette
   entrée disait « laisse les contrastes intacts » : trop fort, corrigé le 2026-09-19.
2. **L'augmentation déplace à peine le niveau, mais écrase les écarts.** +0,9 % sur le niveau.
   L'écart sphere → half, graine par graine, à N = 512 sur le pool d'entraînement : +3,2 /
   +3,6 / +5,0 sur vues augmentées contre +6,3 / +7,1 / +8,5 sur images propres, divisé par
   deux dans chacune des trois graines. Les deux bras se touchent presque sur vues augmentées
   (plus petit half moins plus grand sphere : +0,9) et sont nettement séparés sur images
   propres (+5,6).
3. **Le split ne fait rien**, ni au niveau ni aux écarts. Or les 512 images de la mesure
   augmentée et celles de la mesure propre ne sont pas les mêmes : que train et test, qui
   n'ont aucune image en commun, donnent les mêmes contrastes suggère que l'identité des
   images ne compte pas ici, et que le point 2 vient de l'augmentation elle-même.
4. Conséquence pour le contraste de A-002 : le mécanisme qui pouvait l'expliquer comme un
   artefact de N n'est pas celui qu'on observe. Celui qu'on observe, l'augmentation qui écrase
   les écarts, va dans le sens de ce que ce contraste affirmait. Ce n'est pas une confirmation
   (autres bras, autre budget) : cela affaiblit l'objection la plus probable, sans l'écarter
   pour la paire de A-002.

**Ce que cette relecture NE décide PAS :**
- Rien sur gram contre gram_vicreg ni sur 1 500 pas : les bras relus sont ceux de A-004. Le
  contraste de A-002 reste non vérifié sur sa propre paire, tant que les checkpoints manquent.
- Une seule paire de vues augmentées gelée par graine (v1 seulement), une seule politique
  d'augmentation (forte), 3 graines.
- Une seule famille de bras (sphère), des rangs de 360 à 370 sur un plafond de 512 à N = 512 :
  rien sur des modèles aux rangs très différents, ni proches du plafond, où N = 512 pourrait
  comprimer les écarts.
- Centré seulement. Le contraste de A-002 portait sur le rang non centré, dont la valeur à N = 512 sur
  images propres n'est loggée nulle part.
- Pourquoi l'augmentation écrase les écarts : non mesuré. Hypothèse la plus simple, la variance
  intra-image injectée par l'augmentation, identique entre bras, noie une partie de la
  différence de géométrie ; rien ici ne la teste.
- Post-hoc : aucune prédiction pré-enregistrée, aucun seuil. Descriptif.

**Envoyé à quelqu'un ?** non.

---

### [E-005]  2026-09-16  :  E0d / pilote COCO-Stuff, trou 32 px > rf 23 px, graine 0

```
Commande exacte     : python xp/xp_E0d_coco.py --seed 0 --outdir /content/ubergang_local/results/E0d
                      --data-root /content/ubergang_local/data/cocostuff ; rsync -a ... (Drive)
Graines             : 1 (pilote). Les graines 1 et 2 étaient conditionnées à un régime peuplé
                      (E-004) : elles ne seront pas lancées, le régime est vide.
Budget              : 10 000 pas / bs 128 / 1 passe encodeur ; 5 000 images val2017 ;
                      25 % caché par super-blocs de 32 px, 1 groupe sur 4
Wall-clock          : ~45 min          GPU : A100 (Colab), device=cuda vérifié dans la bannière
Reprise             : run unique. CSV sans trou (200 lignes, pas de 50). Données COCO
                      retéléchargées sur VM neuve avant le run (règle 5). Artefacts archivés
                      dans results/E0d/, copies vérifiées.
Commit ou date des scripts : xp_E0d_coco.py du 2026-09-16 (PREREG_E0D, E-004, écrit avant le run)
```

**Prédiction testée :** P0d, `PREREG_E0D`, enregistrée en E-004 avant le run. Voir cette entrée
pour l'énoncé, la précondition (≥ 10 % des cellules avec `acf_at_rf` > 0,05) et les falsifieurs.

**Résultat brut (graine 0, 1 024 images, 236 385 cellules étiquetées) :**

| statistique | AUC thing/stuff |
|---|---|
| **acf_at_rf (primaire)** | **0,484** |
| témoin naïf, acf_energy | 0,504 |
| **témoin prédicteur nul** | **0,447** |
| acf_at_rf, cœur aveugle (profondeur ≥ 2) | 0,463 |
| acf_at_rf, anneau (profondeur 1) | 0,518 |
| ell_acf | 0,476 |
| ell_freq | 0,522 |
| magnitude du résidu r | 0,551 |

Régime : **8,95 %** des cellules au-dessus de 0,05 (plancher 10 %), 6,99 % thing contre 9,89 %
stuff. Prédicteur nul : 23,97 %. `ell > rf` : 0,00 %. Censure 0 %.
`acf_at_rf` : moyenne **−0,031** (thing −0,0325, stuff −0,0302), écart-type 0,067 ; E0c avait
+0,000 et 0,022. Prédicteur nul : moyenne +0,016, écart-type 0,076.
`ell` moyen 5,06 px (E0c 1,57) ; r moyen 0,170 (E0c 0,0485). Perte : 0,207 (50 à 500) → 0,184
(1 000 à 2 000) → 0,170 (9 000 à 10 000) ; gain de 10 % à 20 % du budget 2,87 %, fusible non
déclenché ; baisse du pas 2 000 à la fin 7,5 % (E0c 17 %).

**Verdict : NO-GO, régime vide (8,95 % contre un plancher de 10 %). Verdict câblé :
« non concluante », raison `regime_empty`. Seuil non touché, et il ne le sera pas.**

Le pré-enregistrement disait « si le régime sort à 8 %, ce sera vide et on le dira tel quel ».
Il est sorti à 8,95 %. Deux graines de plus coûteraient 1 h 30 pour déplacer une moyenne autour
d'un plancher, sur une statistique dont le point 3 ci-dessous montre qu'elle est contaminée :
elles ne seront pas lancées.

**1. Le mécanisme visé a fonctionné.** Le modèle extrapole enfin. Résidu moyen par position dans
le super-bloc de 4 x 4 cellules :

```
0,120  0,177  0,177  0,119
0,167  0,253  0,253  0,167
0,163  0,247  0,247  0,163
0,113  0,165  0,163  0,111
```

Le centre du trou porte 2,2 fois l'erreur des coins, la perte est 3,6 fois celle de E0c (0,170
contre 0,047), et `ell` passe de 1,6 à 5,1 px. Le diagnostic de E-003 était juste : avec un trou
plus grand que le champ réceptif, le modèle ne peut plus interpoler. Ce qui suit n'est donc pas
« le design n'a pas pris », c'est « le design a pris et la réponse est non ».

**2. La séparation est au hasard, et ce qui en dévie va dans le sens inverse de T4.** 0,484 sur
la primaire, 0,463 dans le cœur aveugle, là où le mécanisme est le plus fort. Le témoin
prédicteur nul est à 0,447 : l'image elle-même est plus lisse à 23 px du côté stuff (ciel,
herbe, mur) que du côté thing. Le peu de cohérence longue portée qui existe est une propriété
des images, pas de l'échec du modèle, et elle marque les fonds, pas les objets. C'est
exactement le confondant que le témoin nul avait été pré-enregistré pour attraper.

**3. Le garde-fou de régime, lui, mesurait un artefact de masque.** Décomposition de la variance
de `acf_at_rf` sur les cellules étiquetées :

| facteur | part de la variance de acf_at_rf | idem sur le prédicteur nul |
|---|---|---|
| profondeur de trou | **8,92 %** | 0,75 % |
| position dans le super-bloc | 1,26 % | 0,39 % |
| étiquette thing/stuff | **0,0245 %** | 0,7321 % |

La géométrie du trou explique 360 fois plus de variance que l'étiquette. Le champ du prédicteur
nul, calculé sans masque, ne porte pas cet effet (0,75 %) et porte trente fois plus d'information
thing/stuff (0,73 %) que le résidu entraîné. Signe direct : la moyenne de `acf_at_rf` est passée
de 0,000 (E0c) à **−0,031**, avec un écart-type triplé, alors que le champ nul reste à +0,016.
Un résidu dont l'erreur croît vers le centre de trous de 32 px répartis sur une grille de 32 px
porte une structure périodique imposée ; lue à 23 px, elle est anti-corrélée. **Autrement dit,
si le régime avait franchi 10 %, il aurait été franchi par la géométrie du masque et non par
l'image.** C'est le résultat le plus utile du run, et il n'était pas prévisible depuis E0c.

**4. Ce que la série E établit maintenant, en négatif.** Trois designs, trois régimes vides, par
trois causes distinctes et à chaque fois identifiées : prédicteur myope (E0b), interpolation sans
extrapolation (E0c), et ici extrapolation réelle mais statistique dominée par la géométrie du
masque, avec une séparation au hasard et un témoin d'image qui pointe dans l'autre sens. Le
critère « résidu corrélé au-delà de ce que le modèle voit » n'a jamais pu se déclencher sur ce
banc, et la dernière tentative montre pourquoi c'est difficile en principe : pour forcer
l'extrapolation il faut un masque structuré, et un masque structuré imprime sa propre portée
dans le résidu.

**Ce que ce run NE décide PAS :**
- **Rien sur T4.** Régime vide, pour la troisième fois. Ni « le critère est faux », ni « il est
  vrai ». Ne pas citer 0,484 comme une réfutation.
- Une graine, un jeu (COCO val), une résolution, une architecture, un niveau de résidu, une
  taille de trou. Le point 2 (signe inversé) est descriptif : le falsifieur de signe était
  conditionné à un régime peuplé, il n'a pas tiré.
- Le point 3 est une décomposition de variance, pas un test : elle montre que la géométrie
  domine, elle ne quantifie pas ce que vaudrait la statistique une fois détendancée.
- Rien sur ce que donnerait un masque non structuré (bruit par pixel) : il ne forcerait pas
  l'extrapolation, c'est le compromis identifié au point 4.

**Suite proposée, à trancher par François (aucune option n'est lancée) :**
- **(a) Arrêter la série E** et écrire les trois régimes vides comme un résultat méthodologique
  négatif : « voici trois façons dont ce critère ne peut pas être mesuré, et la tension de
  principe entre forcer l'extrapolation et ne pas imprimer sa propre portée ». Coût 0 h GPU.
  C'est l'option recommandée : les heures A100 valent plus sur B (débloqué) et C2/C3.
- **(b) Un E0e** qui retire la géométrie avant de lire l'ACF (soustraire, par position dans le
  super-bloc et par profondeur, le profil moyen du résidu sur le jeu), précondition et seuils
  refixés avant le run depuis un nouveau calibrage. Coût 45 min par graine. Espérance faible :
  l'étiquette explique 0,02 % de la variance, retirer un nuisible à 9 % ne fabrique pas un
  effet, et le témoin nul dit déjà que la direction est mauvaise. À ne faire que si la question
  de la mesurabilité vaut par elle-même.
- Le checkpoint est resté sur la VM (exclu du rsync) : une ré-analyse de (b) sans réentraîner
  n'est pas possible.

**Envoyé à quelqu'un ?** non. À retenir si ce résultat est présenté un jour : le NO-GO de la
série E est un NO-GO de mesurabilité, pas de séparation.

---

### [E-004]  2026-09-16  :  E0d / pré-enregistrement, COCO-Stuff 128 px, trou 32 px > rf 23 px (run fait, voir E-005)

```
Commande exacte     : python xp/xp_E0d_coco.py --seed 0 --outdir /content/ubergang_local/results/E0d
                      --data-root /content/ubergang_local/data/cocostuff ; rsync -a ... (Drive)
Graines             : 0 d'abord (pilote). 1 et 2 seulement si le régime de la graine 0 est peuplé
                      (règle écrite ici, avant le run : sur régime vide, elles ne décideraient rien).
Budget              : 10 000 pas / bs 128 / 1 passe encodeur ; 5 000 images val2017 ; 25 % caché
                      par super-blocs de 32 px (1 groupe sur 4, chaque super-bloc caché une fois
                      au balayage)                                budget apparié : sans objet
Wall-clock prévu    : ~45 min par graine        GPU : A100 (Colab), vérifier device=cuda
Reprise             : run unique, voir E-005
Commit ou date des scripts : xp_E0d_coco.py du 2026-09-16 (PREREG_E0D), smoke --cpu passé le
                      2026-09-16
```

**Prédiction testée (écrite AVANT le run) :** P0d, `PREREG_E0D`. Avec des blocs cachés de 32 px
plus grands que le champ réceptif du prédicteur (23 px), donc un cœur aveugle d'environ 10 px où
le modèle doit extrapoler, `acf_at_rf` (ACF normalisée du résidu lue à 23 px) est plus élevée sur
les cellules thing que sur les cellules stuff, et les sépare mieux que (i) le naïf à courte
portée `acf_energy` et (ii) la même `acf_at_rf` lue sur un prédicteur nul (écart quadratique de
l'image à sa propre couleur moyenne).

**Précondition (écrite AVANT le run) :** au moins 10 % des cellules étiquetées avec
`acf_at_rf` > 0,05. Seuils tirés de `results/E0c/calib_ruler_synthetic.txt` (grain seul :
q95 = 0,022 ; grain + enveloppe 16 px : environ 13 % au-dessus de 0,05) et de E0c lui-même
(2 % au-dessus de 0,05, régime vide). Sous 10 % : « régime vide », NO-GO par construction,
rapporté à part. `ell > rf` est rapporté pour continuité, sans rôle de garde (E-003, point 4).

**Ce qui la falsifierait (écrit AVANT le run) :** régime peuplé et AUC(acf_at_rf) < 0,60 sur
3 graines (« infirmée ») ; ou AUC ≤ 0,40 (« infirmée, signe inversé » : la statistique suit la
lissité de l'image, pas ce que le modèle n'a pas vu) ; ou marge < 0,03 sur l'un des deux
témoins (« non concluante, pas de marge sur les témoins »). GO seulement si AUC ≥ 0,65,
marge ≥ 0,03 sur chaque témoin et moyenne − écart-type ≥ 0,60, sur 3 graines.

**Fusible :** arrêt à 20 % du budget si la perte a gagné moins de 1 % depuis 10 %.

**Secondaires déclarées :** `ell_acf`, `ell_freq` (attendues épinglées au grain, gardées pour
la comparaison avec E0b et E0c), magnitude du résidu r (0,615 en E0c, descriptif jusqu'ici).
Exploratoires, non gatées : AUC sur le cœur aveugle (profondeur de trou ≥ 2 cellules) contre
l'anneau (profondeur 1), et sur les cellules « stuck » (lam sous la médiane).

**Résultat brut :** run effectué le 2026-09-16, graine 0. Les nombres et la lecture sont dans
l'entrée **E-005** ci-dessus ; cette entrée-ci reste le pré-enregistrement, inchangé depuis.

**Verdict :** NO-GO, régime vide (8,95 % contre un plancher de 10 %). Voir E-005. Le seuil
n'a pas été touché après les données.

**Ce que ce run NE décide PAS (écrit AVANT) :**
- Rien au-delà de COCO val à 128 px, d'un conv-net de 0,37 M paramètres, d'un niveau de résidu
  (pixel) et d'une taille de trou (32 px). Un GO dirait « le critère de T4 est mesurable à
  cette échelle », pas « T4 est vraie ».
- Un « signe inversé » ne réfuterait pas T4 : il dirait que ce design lit l'image et non
  l'erreur ; le témoin nul est là pour le distinguer, pas pour le trancher seul.
- Le seuil 0,05 et le plancher 10 % viennent d'un calibrage synthétique à une seule amplitude
  d'enveloppe ; ils sont fixés ici et ne seront pas retouchés après les données.

**Envoyé à quelqu'un ?** non.

---

### [E-003]  2026-09-16  :  E0c / pilote COCO-Stuff, 128 px, rf 23 px, graine 0

```
Commande exacte     : python xp/xp_E0c_coco.py --seed 0 --outdir /content/ubergang_local/results/E0c
                      --data-root /content/ubergang_local/data/cocostuff ; rsync -a ... (Drive)
Graines             : 1 (pilote ; le verdict pré-enregistré exige 3)
Budget              : 10 000 pas / bs 128 / 1 passe encodeur ; 5 000 images val2017 ; 33 % caché
Wall-clock          : 36 min (2 180 s, 590 img/s ; l'estimation était 2 à 3 h)   GPU : A100 (Colab), device=cuda
Reprise             : run unique. CSV sans trou (200 lignes, pas de 50). Artefacts archivés dans
                      results/E0c/ (summary, CSV, regions.npz) depuis Downloads, copies vérifiées.
Commit ou date des scripts : xp_E0c_coco.py du 2026-09-16 (PREREG_E0C, enregistré avant le run)
```

**Prédiction testée (écrite avant le run) :** P0c, `PREREG_E0C`. Avec un prédicteur dont le champ
réceptif (23 px) dépasse le bloc caché (8 px) et ses voisins, l'ACF normalisée du résidu lue à la
distance du champ réceptif (`acf_at_rf`) sépare thing de stuff mieux que la statistique naïve à
courte portée. Précondition : au moins 5 % des cellules étiquetées avec `ell_acf` brut > 23 px,
sinon « régime vide ». Fusible : arrêt à 20 % du budget si la perte a gagné moins de 1 % depuis 10 %.

**Ce qui la falsifierait :** AUC(acf_at_rf) < 0,60 sur 3 graines, ou marge sur le naïf < 0,03.

**Résultat brut (graine 0, 1 024 images, 262 144 cellules, 236 385 étiquetées : 29 % thing,
61 % stuff, 10 % bord) :**

| statistique | AUC thing/stuff | E0b (E-002) |
|---|---|---|
| **acf_at_rf (primaire)** | **0,468** | s.o. |
| naïf, acf_energy | 0,562 | 0,514 |
| ell_acf | 0,562 | 0,511 |
| ell_freq | 0,570 | 0,517 |
| ell_acf_smooth (biaisé par construction) | 0,507 | s.o. |
| magnitude du résidu r | **0,615** | 0,568 |
| r, bords contre étiquetées | **0,654** | 0,623 |

Régime : **0,00 %** des cellules avec `ell_acf` > 23 px (0 sur 236 385 ; 0 au-delà de 11 px,
0,9 % au-delà de 4 px). `mean_ell` thing 1,61 px, stuff 1,53 px (E0b : 3,23 / 3,21). Censure 0 %.
`acf_at_rf` : thing −0,0015 ± 0,021, stuff +0,0006 ± 0,022, bords −0,0010 ; quantiles 5 et 95 %
à ±0,035 dans les trois classes ; 2 % des cellules au-dessus de 0,05, 0,03 % au-dessus de 0,10.
Magnitude r : bords 0,078 > thing 0,058 > stuff 0,039. corr(ell_acf, acf_energy) = 0,935
(E0b : 0,87) ; corr(acf_at_rf, r) = −0,10 ; corr(acf_at_rf, ell_acf_smooth) = 0,46.
Perte, moyenne par fenêtre de pas : 0,0745 (50 à 500), 0,0598 (500 à 1 000), 0,0562 (1 000 à
2 000), 0,0520 (3 000 à 5 000), 0,0491 (7 000 à 9 000), 0,0466 (9 000 à 10 000). Fusible : gain
de 10 % à 20 % du budget = 6,2 %, non déclenché. lam (progrès entre les pas 7 500 et 10 000) :
médiane +0,018, AUC thing/stuff 0,48.

**Verdict : NO-GO, régime vide (lecture pilote, une graine). Verdict câblé : « non concluante »,
raison `regime_empty`.**

1. **Le prédicteur n'est plus myope, et il apprend jusqu'au bout.** La perte baisse encore de
   17 % entre le pas 2 000 et la fin, contre un plateau dès le pas 1 000 dans E0b. La cause 1 de
   E-002 est levée. Le régime reste vide : la myopie n'était pas la seule cause.
2. **Le résidu est devenu plus blanc, pas plus long.** `ell` passe de 3,2 px (E0b) à 1,6 px. Un
   prédicteur qui voit deux blocs visibles de chaque côté d'un bloc caché de 8 px interpole tout
   sauf le grain fin ; ce qui reste est une erreur d'interpolation à 1,5 px, la même sur un visage
   et sur de l'herbe. Entre E0b (voit trop peu : résidu = image moins un flou, grain à 3 px) et
   E0c (voit assez : résidu = grain d'interpolation à 1,5 px), `ell` a baissé. Aucun réglage entre
   les deux ne produit de portée longue : c'est la production du résidu qui est en cause. Tant que
   la région cachée est plus petite que le champ réceptif, le modèle n'a jamais à extrapoler, et
   une erreur cohérente sur plusieurs blocs (le seul mécanisme qui corrèle le résidu au-delà de
   rf) n'a pas d'occasion de se produire.
3. **La règle `acf_at_rf` lit zéro partout, et on sait à quoi ressemblerait « non zéro ».**
   Calibrage a posteriori de l'instrument sur des champs synthétiques (`xp/calib_ruler_E0c.py`,
   sortie dans `results/E0c/calib_ruler_synthetic.txt`, ni données ni modèle) : grain seul à
   1,5 px, acf_at_rf −0,003 ± 0,016 ; grain modulé par une enveloppe cohérente à 16 ou 32 px,
   +0,022 à +0,029 (q95 0,064 à 0,071). E0c est au niveau du grain seul (−0,002 / +0,001,
   ± 0,022). L'AUC 0,468 sous 0,5 est un déplacement de 0,002 sur une statistique posée à son
   plancher (d ≈ 0,1) ; ce n'est pas un effet inversé.
4. **La précondition de régime était mal posée, et ce run le prouve.** Elle lit `ell_acf` brut,
   le premier passage sous 1/e, que E-002 avait déjà identifié comme épinglé à l'échelle la plus
   fine. Sur le calibrage, un champ portant une enveloppe à 32 px donne encore **0 %** de cellules
   avec `ell` > 23 px : le grain fixe le premier passage quelle que soit la structure longue.
   Aucun résidu contenant du grain ne pouvait la satisfaire. Elle doit être redéfinie sur
   `acf_at_rf` (fraction de cellules au-dessus du plancher du grain) dans tout E0d, avant le run.
   Sur E0c, cette lecture alternative donne aussi un régime vide (2 % au-dessus de 0,05, contre
   environ 13 % attendus sous l'enveloppe synthétique) ; la conclusion ne change pas, l'instrument
   qui la porte change.
5. **`ell` et le naïf sont plus que jamais la même statistique** (corr 0,935). Leur AUC commune
   à 0,56 est le grain légèrement plus grossier des things (1,61 contre 1,53 px), rien de plus.
6. **Ce qui sépare, c'est la magnitude, et elle sépare mieux qu'en E0b.** r : 0,615 thing/stuff
   (0,568), 0,654 bords/étiquetées (0,623). Un meilleur prédicteur creuse l'écart de prévisibilité
   entre stuff et thing, et les bords restent les cellules les moins prévisibles. Statistique non
   pré-enregistrée pour E0c : descriptive.
7. `ell_acf_smooth` : 10,5 px dans les trois classes, AUC 0,507. Le lissage à 13 px impose sa
   propre longueur, comme annoncé. À retirer des versions suivantes.

**Ce que ce run NE décide PAS :**
- **Rien sur T4** : régime vide par construction, comme E-002. Le critère « corrélé au-delà de ce
  que le modèle voit » n'a été testé sur rien.
- Une graine. Lancer les graines 1 et 2 ne changerait pas un régime à 0,00 % (E0b : 0,2 %) ; à ne
  pas lancer sur ce design.
- Un jeu, une résolution, une architecture, un seul niveau de résidu (pixel, `levels=1`).
- Le point 6 (magnitude) est descriptif : rien n'a été pré-enregistré sur r.
- Le calibrage du point 3 est synthétique, à une seule amplitude d'enveloppe, lancé après avoir
  vu les données ; il calibre l'instrument, il ne teste rien. Il a tourné 308 s sur le portable
  (3,3 Go) : plus lourd que prévu, à relancer sur Colab si besoin, jamais localement.

**Ce qu'une version E0d devrait changer, à pré-enregistrer avant tout run :**
- Production du résidu : région cachée plus grande que le champ réceptif (par exemple un bloc de
  32 à 48 px par image), pour que le modèle extrapole. Risque à écrire dans la prédiction : au
  centre d'un tel bloc, l'erreur est l'image moins un a priori, et sa portée est celle de l'image
  elle-même (longue sur du ciel, courte sur une texture), ce qui peut inverser le signe attendu
  par T4. Alternative : lire le résidu au niveau 1 ou 2 (`levels`), où le grain d'interpolation
  est moyenné.
- Précondition sur `acf_at_rf` (fraction de cellules au-dessus du plancher du grain, seuil fixé
  à partir du calibrage, avant le run), et non sur `ell_acf`.
- Retirer `ell_acf_smooth`. Garder la magnitude r comme secondaire déclarée.
- Coût : 36 min A100 par graine à ce design, pas 2 à 3 h.

**Envoyé à quelqu'un ?** non.

---

### [E-002]  2026-09-16  :  E0b / pilote COCO-Stuff, 128 px, graine 0

```
Commande exacte     : python xp/xp_E0b_coco.py --seed 0 --outdir /content/ubergang_local/results/E0b
                      --data-root /content/ubergang_local/data/cocostuff ; rsync -a ... (Drive)
Graines             : 1 (pilote ; le verdict pré-enregistré exige 3)
Budget              : 10 000 pas / bs 192 / 1 passe encodeur ; 5 000 images val2017 (~384 époques)
Wall-clock          : ~55 min                     GPU : A100 (Colab). Un premier lancement sur
                      une session sans GPU a été interrompu ; garde-fou ajouté au script depuis.
Reprise             : run unique. Artefacts dans results/E0b/ (summary, CSV, regions.npz).
Commit ou date des scripts : xp_E0b_coco.py du 2026-09-06 (PREREG_E0B), xp_E_router.py corrigé
```

**Prédiction testée (écrite avant le run) :** P0b, `PREREG_E0B`. Sur COCO à 128 px, la longueur
de corrélation `ell` du résidu sépare les cellules thing des cellules stuff (étiquettes humaines
par région, même image), mieux que la statistique naïve « résidu structuré ». Règle GO/NO-GO
identique à E0, évaluée par `xp_E_router.e0_verdict`.

**Ce qui la falsifierait :** AUC(ell) < 0,60, ou marge sur le naïf < 0,03.

**Résultat brut (graine 0, 1 024 images, 262 144 cellules, 236 385 étiquetées) :**

| statistique | AUC thing/stuff |
|---|---|
| ell_acf | 0,511 |
| ell_freq (proxy LF/HF) | 0,517 |
| naïf, acf_energy | 0,514 |
| energetic (décile bas retiré) | ell 0,513 / naïf 0,514 |
| **magnitude du résidu r** | **0,568** |
| r, bords (cellules mixtes) contre étiquetées | **0,623** |

`mean_ell` thing 3,23 px, stuff 3,21 px ; censure 0 %. Cellules avec `ell` > champ réceptif
(11 px) : **0,18 % thing, 0,29 % stuff**. Perte : 0,202 (pas 50) → 0,165 (pas 1 000) → plateau
jusqu'à 10 000 (0,161). corr(ell, acf_energy) = 0,87 ; corr(ell, r) = 0,03.

**Verdict : NO-GO (lecture pilote, une graine). Interprétable, contrairement à E-001 :**

1. **L'instrument est vivant.** La magnitude du résidu sépare thing de stuff (0,568) et surtout
   les cellules de bord des cellules pleines (0,623 ; bords 0,184 > thing 0,153 > stuff 0,123).
   Un désalignement cellules/étiquettes donnerait 0,50 sur ces deux-là. L'histogramme des ids
   décodé avec la table COCO-Stuff (person, tree, sky, grass en tête) confirme la frontière.
2. **Le régime « ell > champ réceptif » est vide** : une cellule sur 400, et légèrement plus côté
   stuff que côté thing. Le critère de T4 ne peut se déclencher sur rien. Ce run ne dit donc pas
   « ell ne sépare pas », il dit « à cette échelle, rien n'est corrélé au-delà de ce que le
   modèle voit ».
3. **Cause 1, le prédicteur est myope.** Champ réceptif 11 px, bloc caché 8 px, 75 % de masquage :
   il ne voit qu'un anneau d'un pixel et demi autour de chaque bloc, lui-même masqué trois fois
   sur quatre. Il apprend un flou local en 500 pas puis plus rien (10 000 pas étaient vingt fois
   trop longs pour ce modèle). Son résidu est « l'image moins un flou », partout, avec une
   fluctuation fine à ~3 px (grain, texture, contours) que thing et stuff possèdent également.
4. **Cause 2, l'estimateur est épinglé à l'échelle la plus fine.** Le premier passage sous 1/e de
   l'ACF radiale mesure la fluctuation la plus rapide de la fenêtre ; tout bruit haute fréquence
   fixe `ell` à ~3 px quelle que soit la structure longue. corr(ell, acf_energy) = 0,87 : à ces
   échelles, `ell` et le critère naïf sont le même critère, et « battre le naïf » était
   structurellement impossible. `ell_freq` (LF/HF) était l'alternative prévue et lit aussi au
   hasard : le spectre du résidu est indépendant de la classe, ce qui est attendu sous un modèle
   qui ne prédit rien au-delà du flou.
5. Conséquence pour E0 (64 px, `xp_E_router.py --stage e0`) : même champ réceptif (9 px), même
   bloc, même masquage. Le relancer donnerait très probablement le même nul. À ne pas relancer.

**Ce que ce run NE décide PAS :**
- **Rien sur T4** : la thèse dit « structuré au-delà de ce que le modèle voit » ; un modèle qui ne
  voit rien produit un résidu sans portée, et le test est vide par construction, pas négatif.
- Une graine : la lecture pilote n'est pas le verdict. Les graines 1 et 2 n'ont pas été lancées
  (tripler 0,51 sans le comprendre coûtait 3 à 5 h).
- Un jeu (COCO val), une résolution, une architecture.

**Ce qu'une version E0c devrait changer, en pré-enregistrant avant :** champ réceptif qui voit
vraiment au-delà du bloc (profondeur 10, ~23 px) avec fenêtre 64 px (censure 32) ; masquage à
50 % ; un critère de **précondition** rapporté séparément (fraction de cellules avec ell > rf ;
sous 5 %, « régime vide » et non « ell échoue ») ; un troisième estimateur mesuré sur le résidu
lissé à l'échelle du champ réceptif, pour que le grain n'épingle plus la longueur de
corrélation ; et un arrêt si la perte plafonne avant 10 % du budget (modèle trop faible).

**Envoyé à quelqu'un ?** non. Ce NO-GO est un NO-GO de régime vide, pas de séparation : à ne pas
présenter comme un résultat négatif sur T4.

---

### [E-001]  2026-08-22  :  E0 / diagnostic GO/NO-GO, 64 px, 3 graines (entrée écrite a posteriori le 2026-09-16)

```
Commande exacte     : python xp/xp_E_router.py --stage e0 (défauts : 64 px, patch 8, depth 3,
                      15 000 pas, bs 256, STL-10 unlabeled 20 000 + DTD 1 880)
Graines             : 3 (0, 1, 2)
Budget              : 15 000 pas / bs 256 / 1 passe encodeur       budget apparié : sans objet
Wall-clock          : ~45 min par graine        GPU : A100 (Colab)
Reprise             : run unique. Artefacts : zip E du Drive (summaries, CSV, figures).
Commit ou date des scripts : xp_E_router.py du 2026-08-21, AVANT correctif
```

**Résultat brut :** `auc_ell_acf`, `auc_ell_freq`, `auc_naive` = **NaN** sur les trois graines ;
`frac_object_regions` = 1,0 ; `mean_ell_texture` = NaN ; `mean_ell_object` ≈ 3,15 px.

**Verdict : NO-GO affiché, sans valeur.** Panne d'instrument : le jeu de diagnostic prenait les
1 024 premières images d'un pool ordonné objets-puis-textures, donc 1 024 objets et 0 texture ;
classe négative vide, AUC NaN, verdict retombé en NO-GO par `NaN < 0,60`. La fonction de tirage
stratifié existait dans le fichier et n'était jamais appelée. Corrigé le 2026-08-26 (tirage
stratifié partagé par les deux fonctions de mesure, garde-fou de reprise, arrêt immédiat si une
seule origine). Jamais relancé : E-002 montre que le design partage un défaut plus profond
(champ réceptif 9 px, `mean_ell_object` déjà à 3,15 px) et le remplace.

**Ce que ce run NE décide PAS :** rien, sur rien. Ne pas citer son NO-GO.

---

### [A-004]  2026-09-11  :  A2 / reprise à 4 bras (sg, sphère, sphère_gamma, sphère_gamma_half), 6 000 pas

```
Commande exacte     : python xp/xp_A_diagnostics.py --exp a2 --arm gram_vicreg_sg
                      --arm gram_vicreg_sphere --arm gram_vicreg_sphere_gamma
                      --arm gram_vicreg_sphere_gamma_half --clip 5.0 --steps 6000
                      --outdir /content/ubergang_local/results/A2v2_repair ; rsync -a ... (Drive)
Graines             : 3 (0, 1, 2). Init identique à A-003 (rang propre centré à l'init
                      238,8 / 200,3 / 218,6 dans les deux runs).
Budget              : 6000 pas / bs 256 / 2 passes encodeur (sg : +2 passes MLP déclarées)
                      budget apparié : oui, 6000 pas mesurés partout
Wall-clock          : 747 à 758 s par run, ~2,5 h        GPU : A100 (Colab)
Reprise             : run unique, resumed_from_step = 0 partout. Sauvé par le rsync en fin de
                      cellule, téléchargé dans Downloads, archivé dans results/A2v2_repair/
                      (A_summary.json, 12 CSV, 11 summaries par run : celui de sphere seed2 n'a
                      pas été téléchargé, ses valeurs sont dans A_summary.json ; le CSV de
                      sphere_gamma_half seed1 a été re-téléchargé, la première copie s'arrêtait
                      au pas 1250).
Commit ou date des scripts : xp_A_diagnostics.py du 2026-09-09
```

**Prédiction testée (écrite avant le run) :** Q1 et Q2 de `PREREG_A2V2`, inchangées, registrées le
2026-09-05. `gram` et `gram_vicreg` ne sont pas dans ce run : les verdicts utilisent la baseline
`gram` de A-003, même config, mêmes graines, même init. Le bras `sphere_gamma_half`
(gamma = 0,5/√d) n'est **pas** pré-enregistré : ajouté le 2026-09-09, après la falsification de
Q2, pour tester mon auto-critique de A-003 (gamma = 1/√d est le plafond, donc pas satisfiable).
Sa lecture est exploratoire.

**Ce qui les falsifierait :** Q1, kNN(sg) − kNN(gram) < −1,8 pt. Q2, gain(sphere_gamma) ≥ 60.

**Résultat brut (moyennes sur 3 graines, écart-type inter-graines) :**

| bras | charnière active (z norm.) | kNN % | sonde lin. % | rang backbone propre centré | rang z |
|---|---|---|---|---|---|
| gram (A-003, baseline) | s.o. | 35,71 ± 0,76 | 37,26 ± 0,45 | 312,4 | 8,96 |
| gram_vicreg_sg | s.o. | 33,09 ± 1,29 | 38,48 ± 1,71 | 144,2 (1,8 / 237,2 / 193,7) | 7,3 |
| gram_vicreg_sphere (γ=1) | **1,00** | **44,52 ± 0,89** | **44,50 ± 0,83** | 445,7 ± 0,2 | 123,3 |
| gram_vicreg_sphere_gamma (γ=1/√d) | 0,51 | 41,07 ± 1,93 | 42,08 ± 1,53 | 448,2 ± 2,2 | 123,3 |
| gram_vicreg_sphere_gamma_half (γ=0,5/√d) | 0,11 | 39,31 ± 1,31 | 40,46 ± 0,93 | **454,0 ± 1,4** | 95,9 |

**Verdicts :** Q1 **INFIRMÉE**, −2,62 pt (−3,0 σ), seuil −1,8. Q2 **INFIRMÉE**,
gain(sphere_gamma) = +135,7 (seuil 60), gain(sphere) = +133,3. Le bloc Q du script exige
`gram` dans le run et ne s'est pas exécuté pendant le run ; les verdicts ont été recalculés
après coup en appelant `verdicts_a2` sur les summaries de ce run fusionnés avec les trois
graines de `gram` de A-003 (même code, mêmes seuils).

**Ce que ce run établit, au-delà des verdicts :**

1. **Les charnières, enfin mesurées.** `sphere` : 1,00 à chaque pas. `sphere_gamma` : 1,00 →
   0,84 (250) → 0,64 (2000) → 0,51 (6000) et y reste : elle chevauche le plafond, moitié des
   coordonnées dessus, moitié dessous ; l'auto-critique de A-003 était juste, ce bras n'est pas
   satisfiable. `sphere_gamma_half` : 1,00 → 0,25 en 250 pas → 0,11 stable : **première
   contrainte de variance réellement satisfaite du banc.**
2. **Sur ce bras satisfait, le rang du backbone est le plus haut** (454,0 ; +8,3 sur `sphere`,
   10 σ). Quand la charnière est éteinte, ce qui atteint encore le tronc est le terme de
   covariance sur la sphère, qui n'a pas de charnière. Q2 est morte sur un bras honnête.
3. **Le transfert suit la persistance de la pression de variance, pas la géométrie qu'elle
   laisse.** Charnière active 1,00 / 0,51 / 0,11 → kNN 44,52 / 41,07 / 39,31 (extrêmes : 5,2 pt,
   5,7 σ), sonde linéaire 44,50 / 42,08 / 40,46 (5,6 σ). Rang backbone 445,7 / 448,2 / 454,0 :
   **ordre inverse**. Rang z 123 / 123 / 96, uniformité −3,93 / −3,93 / −1,08 : aucune
   statistique statique loggée n'ordonne les trois bras comme le transfert.
4. **sg : le mécanisme est visible dans les trajectoires.** |z| passe de 1,2 à ~30 en 250 pas
   dans les trois graines (la tête gonfle son échelle pour satisfaire la charnière sans le
   tronc), pics de perte à 1,1e4 (graine 0, pas 2725) et 2,0e4 (graine 1, pas 3950) tenus par le
   clip, rang backbone centré sur vues augmentées 100 à 137 → 1,4 / 21 / 16, rang propre **non
   centré ≈ 3 dans les trois graines** (dérive vers un vecteur moyen géant), effondrement total
   de la graine 0. Laquelle des graines s'effondre change d'un run à l'autre (graine 1 en A-003,
   graine 0 ici). Lecture : variance et covariance de VICReg sont calculées sur z **centré**,
   donc aveugles à la moyenne ; sans le tronc dans la boucle, la tête satisfait la contrainte
   par « grand vecteur moyen + bruit », et le gradient d'accord transmis par ce projecteur gonflé
   effondre le tronc. Le terme de légalité a besoin du tronc pour stabiliser quoi que ce soit.
5. **Reproductibilité inter-runs.** `sphere` à config identique : 43,53 (A-003) puis 44,52 ici,
   +1,0 pt de non-déterminisme GPU. Toute comparaison inter-runs porte cette incertitude ; Q1
   (−2,62 contre −1,8) la survit.

**Ce que ce run NE décide PAS :**
- Rien sur l'échelle : CIFAR, ResNet-18, un lr, un clip partagé qui mord aussi les bras sains.
- Le point 3 est une lecture, pas un test : « persistance de la pression » est confondue avec
  gamma lui-même. Un bras à gamma fixe et pression intermittente (masquage temporel du terme)
  trancherait. L'écart gamma−half seul n'est qu'à 1,3 σ ; gamma a une graine basse (38,9).
- Le bras `_half` est post-hoc : ses chiffres décrivent, ils ne confirment rien.
- `sg` n'est pas un contrefactuel propre : instable, bistable entre graines et entre runs. Q1 est
  infirmée formellement, mais l'information est l'instabilité, pas le −2,62.

**Envoyé à quelqu'un ?** non.

---

### [A-003]  2026-09-06  :  A2 / cinq bras, budget long, deux falsifieurs qui tirent

```
Commande exacte     : xp/xp_A_diagnostics.py --exp a2 --arm gram --arm gram_vicreg
                      --arm gram_vicreg_sg --arm gram_vicreg_sphere
                      --arm gram_vicreg_sphere_gamma --clip 5.0 --steps 6000
Graines             : 3 (0, 1, 2)        budget apparié : oui (matched_budget_check OK,
                      6000 pas x 256 x 2 fwd pour les cinq bras)
Wall-clock          : 11202.9 s (~3.1 h) A100
Reprise             : les 6 runs gram/gram_vicreg pré-existants ont été RÉ-ENTRAÎNÉS,
                      clip venant d'entrer dans la config par run (voulu)
Pré-enregistrement  : PREREG_A2V2 (Q1, Q2), enregistré 2026-09-05 avant tout run
                      des bras sg et sphere_gamma. Seuils NON touchés depuis.
```

**AVERTISSEMENT DE PROVENANCE.** Le outdir local n'a jamais été synchronisé vers le
Drive et la VM Colab a été recyclée avant toute copie. **Tous les artefacts (CSV,
summaries, checkpoints, figures) sont PERDUS.** La seule trace est le stdout, archivé
verbatim dans `results/A2v2_long/RUN_LOG.txt`. Tous les nombres ci-dessous en viennent.
Conséquence : les valeurs *par graine* de kNN, sonde linéaire et rangs propres (init et
final) sont récupérables du log ; les *trajectoires* (charnière, pics de perte, courbe de
décrochage) ne le sont pas. Deux bras seront rejoués pour ça (voir plus bas).

**Le clip 5.0 est un changement d'instrument, pas cosmétique.** À clip 0 le bras sg
diverge (loss=inf au pas 43, cov=inf : coupé du tronc, le projecteur satisfait la charnière
en gonflant son échelle, et le terme de covariance croît en puissance 4 de cette échelle).
Le clip 5.0 stabilise, mais il mord aussi gram (kNN 36.6 sans clip -> 35.7 avec). **Les
chiffres A2v2 se comparent entre eux, jamais aux séries sans clip** (A-001, A-002).

**Résultats finaux, moyennes sur 3 graines (± écart-type inter-graines) :**

| bras | kNN % | sonde linéaire % | rang h propre centré | rang z |
|---|---|---|---|---|
| gram | 35.71 ± 0.76 | 37.26 ± 0.45 | 312.4 | 8.96 |
| gram_vicreg | 33.77 ± 0.14 | 41.37 ± 1.13 | 209.2 | 72.29 |
| gram_vicreg_sg | 33.21 ± 2.54 | 39.67 ± 2.51 | 252.2 (bimodal : 339 / 65 / 353) | ~73 |
| gram_vicreg_sphere | **43.53 ± 0.75** | **43.96 ± 0.60** | **445.9** | 123.4 |
| gram_vicreg_sphere_gamma | 40.11 ± 0.63 | 42.00 ± 1.00 | 448.0 | ~123 |

**Q1 (fuite de gradient) : INFIRMÉE.** kNN(sg) − kNN(gram) = −2.50 pp (seuil d'infirmation
−1.8). Couper le gradient de légalité avant le backbone NE récupère PAS le kNN. Le dommage
ne voyage donc pas par le gradient du terme dans le tronc ; il passe par le projecteur
remodelé, qui change ce que le terme d'accord demande au tronc.
  RÉSERVE À LIRE AVEC LE VERDICT : le bras sg est dynamiquement INSTABLE. Écart-type
  inter-graines 2.54 pp (17x celui de gram_vicreg) ; graine 1 à demi-effondrée (rang centré
  65 vs ~345) ; rang NON centré de 2.3 pour un centré de 339 sur les graines 0/2 (features
  concentrées sur une direction moyenne géante). La participation du tronc au terme de
  légalité est ce qui stabilise le système, ce que la divergence à clip 0 criait déjà.
  Le falsifieur a tiré proprement, mais le bras qui le déclenche n'est pas un bras sain.

**Q2 (saturé vs satisfiable) : INFIRMÉE.** gain de rang backbone : sphere +133.4,
sphere_gamma +135.6. IDENTIQUES. Une cible de variance atteignable remodèle le tronc
autant qu'une cible saturée. L'hypothèse « satisfiable s'arrête à la tête » est fausse :
**l'ingrédient actif est la NORMALISATION de z, pas la valeur de gamma.**
  AUTO-CRITIQUE DE DESIGN (à corriger, ne pas cacher) : gamma_sphere_gamma = 1/sqrt(d) =
  0.0884 est le PLAFOND même du std par coordonnée sur la sphère. Le bras « satisfiable »
  était donc en pratique quasi saturé lui aussi, ce qui a vidé le contraste que Q2 voulait
  créer. Un vrai bras satisfiable mettrait gamma = 0.5/sqrt(d). La falsification est
  formellement valide (le falsifieur portait sur le gain de rang, pas sur gamma), mais elle
  ne tranche pas la question que Q2 croyait poser. -> bras sphere_gamma_half à rejouer.

**LES DEUX VRAIES DÉCOUVERTES DU RUN (au-delà des falsifieurs) :**

1. **Le bras "interdit" domine tout.** À 6000 pas, gram_vicreg_sphere fait 43.53 % kNN et
   43.96 % sonde linéaire : +7.8 et +6.7 pp sur gram, meilleur PARTOUT, graines serrées. À
   1500 pas (A-001) son avance était insignifiante ; à 4x le budget elle est écrasante.
   P2.3b bascule INFIRMÉE (rang z sphère 123 >> 80 % de gram_vicreg). La narration de T2,
   « VICReg sur z normalisé est l'erreur que le garde-fou du socle interdit », est
   contredite par le banc : la config interdite est la MEILLEURE config mesurée. Le
   garde-fou vicreg_reg protège contre un mode d'échec (gradient qui s'annule) qui, à ce
   budget et cette échelle, ne se matérialise pas ; la charnière saturée agit comme une
   pression stationnaire utile. RÉSULTAT DE THÈSE, pas un détail.

2. **kNN et sonde linéaire sont ANTI-corrélés entre bras.** gram_vicreg vs gram : −1.9 pp
   kNN, **+4.1 pp sonde linéaire**. Le "dommage" du terme de légalité n'est pas une
   propriété de la représentation, c'est une propriété du COUPLE représentation-sonde :
   nuisible à un vote cosinus local (kNN), bénéfique à un hyperplan entraîné (linéaire).
   Deux contre-exemples à rang apparié dans ce seul run : (i) gram vs gram_vicreg, rangs z
   très différents mais le point ci-dessus ; (ii) sphere vs sphere_gamma, MÊME rang h
   (446 vs 448) et pourtant 3.4 pp de kNN d'écart (~5 sigma). Le rang n'est un médiateur
   ni suffisant ni fiable du transfert. Touche directement la thèse RankMe (le rang
   effectif comme sélecteur de modèle sans étiquettes).

**Ce que A-003 NE décide PAS :**
- rien sur ImageNet ni au-delà de 6000 pas / ResNet-18 / CIFAR-10 / 32x32 ;
- la mécanique EXACTE derrière la domination de la sphère n'est pas isolée (normalisation ?
  échelle de z ? la charnière comme force constante ?) ; la paire sphere/sphere_gamma
  montre que gamma n'est PAS le levier, elle ne dit pas quel est le levier ;
- l'instabilité du bras sg n'est pas caractérisée dans le temps (artefacts perdus) ;
- une seule pondération (var_w=25, cov_w=1), un seul lr, un seul clip. Le clip lui-même est
  un hyperparamètre qui touche la dynamique mesurée.

**À rejouer (et à SYNCHRONISER cette fois, cf. RUN_LOG.txt) :** gram_vicreg_sg (trajectoire
de l'instabilité, 3 graines) et un nouveau bras gram_vicreg_sphere_gamma_half
(gamma = 0.5/sqrt(d), le vrai bras satisfiable). ~1 h 20 A100. Le reste des conclusions
tient sur les nombres du log.

**Envoyé à quelqu'un ?** non.

---

### [A-002]  2026-08-26  :  A2 / réanalyse des artefacts, aucun GPU

```
Commande exacte     : aucune. Relecture de results/A/A_summary.json (généré le
                      2026-08-22 14:38:35), champ A2_raw.per_arm, 3 bras x 3 graines.
Graines             : 3 (0, 1, 2), celles de A-001
Budget              : inchangé, 1500 pas / bs 256 / 2 passes fwd    budget apparié : oui
Wall-clock          : 0 h            GPU : aucun
Reprise             : sans objet, aucun entraînement
Commit ou date des scripts : xp_A_diagnostics.py du 2026-08-22, non modifié
```

**Statut, à lire avant les chiffres : ceci n'est PAS un run.** L'hypothèse a été formulée
**après** avoir vu A-001, sur ses propres artefacts. Le gabarit exige une prédiction écrite
avant ; il n'y en a pas. Cette entrée est un **diagnostic**, pas un test. Elle ne peut
confirmer aucune thèse. Ce qu'elle peut faire, et fait, c'est **retirer une lecture** de
A-001.

**Ce qui a été vérifié.** `eff_rank_z` est le rang effectif de la sortie du **projecteur**
(128 dim, ligne 835). `knn_acc` est mesuré sur le **backbone** (`knn_probe(model.backbone, ...)`,
ligne 1037, 512 dim). Deux espaces séparés par un MLP à deux couches avec batchnorm.
`eff_rank_h` était déjà mesuré, ligne 836, déjà tracé, ligne 1185, et jamais lu.

**Résultat brut, moyennes sur 3 graines avec écart-type inter-graines :**

| bras | eff_rank_z /128 | eff_rank_h /512 | kNN % |
|---|---|---|---|
| gram | 15,15 ± 2,65 | 146,9 ± 7,8 | 36,40 ± 0,76 |
| gram_vicreg | 73,20 ± 0,86 | 147,1 ± 5,1 | 34,11 ± 0,44 |
| gram_vicreg_sphere | 123,45 ± 0,03 | 202,1 ± 1,3 | 37,97 ± 1,58 |

En erreur standard de la différence, gram vers gram_vicreg : `eff_rank_z` **x4,83**,
`eff_rank_h` **+0,21, soit 0,04 σ**, kNN **−2,29 points, 4,5 σ**.
gram vers sphere : `eff_rank_h` x1,38, **12,1 σ**, kNN +1,57 point à 1,55 σ, non concluant.

**Verdict : l'anomalie rang/kNN de A-001 est retirée telle qu'elle était formulée.** Le
rang multiplié par cinq est celui d'un espace où rien n'est évalué. Le rang de l'espace où
le kNN est mesuré ne bouge pas. « 4,8x le rang, −2,3 points » mettait en regard deux
espaces différents.

**Ce qui reste, et qui est plus net.** À rang de backbone identique (0,04 σ), le terme de
légalité coûte **2,29 points de kNN à 4,5 σ**. Le dommage existe et **n'est pas médié par
le rang**. Le bras sphère, seul à monter le rang du backbone, n'est pas moins bon en kNN.

**Réserve à faire voyager avec P2.2.** P2.2 reste `confirmee` et dit ce qu'elle dit :
gram_vicreg bat gram sur le rang de `z`. Mais son propre bloc de preuves porte
`knn_gram_vicreg` 0,341 contre `knn_gram` 0,364. Énoncer P2.2 sans cette ligne la fait lire
comme « le terme de légalité aide », ce que le seul chiffre aval du run contredit.

**Ce que cette réanalyse NE décide PAS :**
- **Rien sur T2.** Aucune thèse n'est testée ici, l'hypothèse est postérieure aux données.
- **Rien sur la cause** des −2,29 points. Le rang est écarté comme médiateur, rien n'est mis
  à sa place.
- `eff_rank_h_at_init` n'est pas journalisé, donc on ignore si 146,9 est au-dessus ou en
  dessous de l'initialisation. À ajouter avant tout run qui voudrait conclure.
- Les réserves de A-001 tiennent toutes : 1500 pas, kNN de 34 à 38 % pour un hasard à 10 %,
  trois bras sous-entraînés, une seule métrique aval, un seul dataset.

**Envoyé à quelqu'un ?** non. Et **à ne citer nulle part en l'état** : l'anomalie rang/kNN de A-001
est retirée.

---

### [A-001]  2026-08-22  :  A1 + A2 / invariance O(d) et effondrement relationnel

```
Commande exacte     : python xp/xp_A_diagnostics.py --exp all
Graines             : 3 (0, 1, 2)
Budget              : 1500 pas / bs 256 / 2 passes fwd     budget apparié aux bras : oui
Wall-clock          : 53 min          GPU : A100 40GB, torch 2.11.0+cu128
Reprise             : run unique, resumed_from_step=0      CSV sans trou : oui
Dataset             : CIFAR-10, ResNet-18 cifar-stem, proj_dim 128
```

**Prédictions testées / verdicts :** P1.a-d confirmées (T1) · P2.1 **infirmée** ·
P2.2 confirmée · P2.3a confirmée · P2.3b **infirmée**

**Résultat brut :**

| bras | eff_rank /128 | σ/plafond | kNN (3 graines) | align | unif |
|---|---|---|---|---|---|
| gram seul | 15,2 ± 2,2 (12 %) | 10,1 % | 36,4 ± 0,6 % | 0,020 | −0,041 |
| gram + VICReg | 73,2 ± 0,7 (57 %) | 37,5 % | 34,1 ± 0,4 % | 0,271 | −0,561 |
| VICReg sur sphère | 123,4 ± 0,0 (96 %) | 99,9 % | 38,0 ± 1,3 % | 1,962 | −3,934 |

A1, résidu de rotation en float64 : gram 4,0e-19 · InfoNCE 1,1e-16 · protos fixes 1,45e-2 ·
protos ajustés **5,74** · contrôle rotation 3,0e-17 · contrôle permutation 2,9e-17.

**Verdict : T1 confirmée en entier. T2 partiellement infirmée, contre moi.**

**Ce que ce run NE décide PAS :**
- 1 500 pas, kNN ~0,36 : les trois bras sont **sous-entraînés**. Rien sur le comportement à
  convergence, ni à l'échelle ImageNet, ni sur le transfert aval réel (kNN ≠ probe linéaire).
- Le falsifieur de P2.1 (`rang < 3`) était mal choisi : il teste l'effondrement *total* alors
  que le phénomène réel est une concentration partielle, visible en σ/plafond et en uniformité.
  « Infirmée » ici veut dire « pas d'effondrement total », pas « pas de concentration ».
- L'anomalie rang/kNN (gram_vicreg : 4,8× le rang, −2,3 pp de kNN, 4,5 σ groupés) est solide
  statistiquement mais n'est mesurée que sur **une** métrique aval et **un** dataset.

**Envoyé à quelqu'un ?** non.

---

