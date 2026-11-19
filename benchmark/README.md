# Campagnes de benchmark d'inférence

Les workflows **Benchmark CPU** et **Benchmark GPU** exécutent les moteurs
successivement dans un seul job, sur le même runner. Leurs machines peuvent
être différentes : comparer les moteurs au sein d'un même job. Le GPU est
limité au premier GPU visible. Les serveurs locaux sont arrêtés avant le moteur
suivant. Le job GPU partage son groupe de concurrence avec le smoke test GPU.

Au lancement, le menu **model** propose deux profils :

* `qwen2.5-0.5b` (par défaut) compare Magnetar, Candle, llama.cpp, Ollama et
  AirLLM sur `Qwen/Qwen2.5-0.5B-Instruct`. Strata est signalé comme
  indisponible : Qwen2.5 ne fait pas partie de ses modèles documentés.
* `qwen3.6-35b-experimental` compare llama.cpp et, sur GPU, Strata avec le
  même fichier `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` et les mêmes identifiants de
  tokens d'entrée. Magnetar ne prend pas encore en charge cette architecture
  dans sa campagne de qualification ; Candle, Ollama et AirLLM ne sont pas
  intégrés pour ce profil. Ils sont signalés comme indisponibles, et non
  comptés comme des succès. Le profil CPU ne lance pas Strata, dont ce moteur
  nécessite un GPU compatible.

Le profil expérimental télécharge environ 22,9 Go de poids. Prévoir au moins
32 Go de RAM et 30 Go d'espace disque libre, ainsi qu'un GPU NVIDIA et un
pilote compatibles avec le moteur Strata CUDA 13. Les runners CPU GitHub
ordinaires peuvent ne pas disposer d'assez de mémoire. La campagne vérifie la
mémoire disponible et l'espace disque avant le téléchargement ; une machine
insuffisante produit un échec explicite. Ces seuils ne garantissent pas que
les deux moteurs pourront charger le modèle sur chaque machine.

Les choix sont fondés sur la [documentation Qwen3.6 de Strata](https://github.com/Niko1221/Strata/blob/main/docs/QWEN36.md).
Le modèle et le moteur Strata sont téléchargés à des versions fixées et leurs
empreintes SHA-256 sont vérifiées. Le mode expérimental désactive MTP ; il
utilise les experts sur CPU pour llama.cpp et le placement hybride propre à
Strata.

Chaque campagne utilise le prompt brut `The capital of France is`, une
génération gloutonne, un échauffement et, par défaut, cinq requêtes mesurées
d'au plus 16 tokens. `tokens` et `runs` sont configurables au lancement ;
`tokens` doit être supérieur ou égal à 2. Les autres checkpoints et moteurs
tiers utilisent également des révisions fixées.

## Résultats et interprétation

Le résumé du job et l'artefact contiennent `comparison.md`, `status.tsv`,
les informations du matériel, les journaux et un `report.json` avec ses
données brutes par moteur. Un échec n'interrompt pas les moteurs suivants,
mais fait échouer le job ; les résultats partiels restent accessibles.

Les conditions de mesure diffèrent et doivent être examinées avant de classer
les moteurs :

* Magnetar mesure en streaming, AirLLM utilise une API Python synchrone,
  llama.cpp et Ollama une API HTTP locale, et Strata un protocole local à
  tokens. Ces frontières ne sont pas identiques.
* Candle lance un nouveau processus et recharge le modèle à chaque mesure.
  Son exemple ne fournit qu'un débit arrondi incluant le préremplissage et la
  génération ; la latence du tableau est estimée à partir de ce débit. Le temps
  du processus avec chargement figure dans le rapport brut. L'exemple ne
  rapporte pas le nombre de tokens du prompt.
* Pour Qwen2.5, Magnetar et AirLLM utilisent les poids Hugging Face en F32 ;
  Candle utilise F32 sur CPU et BF16 sur GPU ; llama.cpp et Ollama utilisent
  le GGUF FP16. Les précisions et chemins de calcul diffèrent.
* Le cache de prompt de llama.cpp est désactivé. Ollama et Strata peuvent
  réutiliser un préfixe ; le rapport signale les tokens réutilisés, car le
  travail de préremplissage peut alors différer.
* Le tableau indique les nombres de tokens disponibles. Vérifier également
  les sorties, les placements GPU, les versions et les données brutes avant
  toute comparaison. Une occupation VRAM positive ne prouve pas que tout le
  modèle est sur GPU.
* Sur CPU, Magnetar utilise son Provider CPU optimisé (`cpu-native`).

Le scénario court et mono-requête ne permet pas de conclure sur les contextes
longs, la concurrence ou la performance générale. Le processeur d'un runner
CPU GitHub peut changer entre campagnes ; répéter sur une machine stable pour
une étude durable.

## Utilisation locale

La campagne CI Linux peut être lancée depuis la racine dans un environnement
adapté, avec Rust, CMake, git, curl, Python et les dépendances de construction :

```bash
export RUNNER_TEMP="$(mktemp -d)"
export BENCH_DEVICE=cpu BENCH_TOKENS=16 BENCH_RUNS=5
export BENCH_MODEL_PROFILE=qwen2.5-0.5b
bash benchmark/campaign.sh
```

Pour CUDA, choisir `BENCH_DEVICE=gpu` et utiliser un environnement compatible
avec le toolkit, le pilote NVIDIA et PyTorch. Pour le grand modèle, définir
`BENCH_MODEL_PROFILE=qwen3.6-35b-experimental` et respecter les ressources
indiquées ci-dessus. Les campagnes téléchargent des modèles et construisent
des composants ; prévoir suffisamment de disque.

Les adaptateurs historiques restent accessibles individuellement avec
`python benchmark/run.py --help`. Pour comparer leurs rapports individuels :
`python benchmark/compare.py chemin/vers/report1.json chemin/vers/report2.json`.
La suite de tests sans dépendance externe se lance avec
`python -m unittest discover -s benchmark/tests -v`.
