# Campagnes de benchmark d'inférence

Les workflows **Benchmark CPU** et **Benchmark GPU** exécutent successivement
Magnetar, llama.cpp, Ollama et AirLLM dans un seul job, sur le même runner.
Chaque workflow choisit une machine distincte : les résultats CPU et GPU ne
proviennent pas nécessairement du même hôte. Le GPU est limité au premier GPU
visible, et les serveurs llama.cpp et Ollama sont démarrés localement puis
arrêtés avant le moteur suivant. Le job GPU partage son groupe de concurrence
avec le smoke test GPU.

Une campagne utilise `Qwen/Qwen2.5-0.5B-Instruct`, le prompt brut
`The capital of France is`, une génération gloutonne, un échauffement et, par
défaut, cinq requêtes mesurées d'au plus 16 tokens. `tokens` et `runs` sont
configurables au lancement ; `tokens` doit être supérieur ou égal à 2.
Les checkpoints Hugging Face et GGUF sont téléchargés à des révisions fixées
et leurs empreintes sont vérifiées. Les moteurs tiers sont installés ou
construits dans le répertoire temporaire du job ; leurs versions figurent dans
les rapports ou journaux.

Strata n'est pas exécuté dans cette campagne : le modèle Qwen2.5 choisi n'est
pas documenté parmi les modèles pris en charge par Strata. Il figure comme
indisponible dans le tableau. Une comparaison à cinq nécessite d'abord un
modèle effectivement pris en charge par tous les moteurs et par la campagne
Magnetar.

## Référence Strata pour Qwen3.8-Flash-Next

Le workflow **Benchmark GPU** lancé avec `suite: strata-flash-next` mesure
[Strata](https://github.com/Niko1221/Strata) (MIT), à un commit fixé
(`strata_ref`), servant Qwen3.8-Flash-Next sur le runner GPU : la référence
de l'implémentation du modèle dans Magnetar. Le setup de Strata télécharge la
taille ISTA-DASLab choisie (`strata_model`, IQ2_XS par défaut, environ 70 Go
avec la table n-gram) et la place lui-même : les poids actifs à chaque token et
un cache d'experts en VRAM sur les deux cartes, les autres experts en RAM ou
lus depuis le SSD quand la RAM ne suffit pas, la table n-gram sur le SSD.
Contexte 8 192 tokens, sans images.

Le serveur est mesuré deux fois, avec puis sans la couche MTP de drafting
(`"mtp": false` dans sa configuration), Magnetar n'ayant pas de drafting :
une cellule du harness (`benchmark/ci.sh`, le prompt de la campagne) et
`terminal/tools/strata_bench.py` de Strata (questions courtes : décodage et
TTFT ; prompts de 4K et 16K tokens : prefill). L'artefact contient le journal
du setup, la configuration générée, les journaux du serveur, la mémoire GPU et
RAM occupée et les rapports.

### Référence mesurée (run 45446491115, 2026-11-24)

Runner `arc-gpu-magnetar` : 2 × RTX 3060 12 Go (PCIe 3.0), Intel Core
i7-7700 (4 cœurs, 8 threads), 31 Go de RAM, NVMe. Strata d46f5f2, moteur
précompilé 0.1.67 (CUDA 13), Qwen3.8-Flash-Next IQ2_XS d'ISTA-DASLab (deux
fichiers : 39,2 Go de modèle, 28,8 Go de table n-gram), contexte 8 192, cache
KV int8. Téléchargement et préparation : 23 minutes.

Placement choisi par Strata :
- couches réparties entre les deux cartes (`layer split auto`) ;
- experts **non copiés en RAM** : réécrits une fois dans un fichier de 33 Gio,
  puis mappés depuis le SSD (`--mmap-experts`), le cache de pages de l'OS en
  gardant ~24 Go ; RAM utilisée par le processus : 7 à 9 Go ;
- cache d'experts GPU : 6 483 experts sur 24 576 (8,57 Gio de VRAM), taux de
  succès de 87 à 98 % en décodage une fois chaud (73 à 86 % aux premières
  requêtes) ; 11,1 à 11,3 Go de VRAM occupés par carte.

| Mesure (`strata_bench.py`, médiane et étendue) | Sans drafting | Avec drafting (DFlash2) |
| --- | --- | --- |
| Décodage, questions courtes (100 tokens) | 23,3 tok/s (16,1–34,1) | 38,3 tok/s (21,9–43,7) |
| TTFT, prompt de ~46 tokens | 1,15 s (0,74–1,59) | 1,03 s (0,77–1,50) |
| Prefill, prompt de 4 019 tokens | 353 tok/s (268–546) | 369 tok/s (246–513) |

- Le setup a choisi le drafter DFlash2 (1,21 Go) plutôt que la couche MTP du
  modèle ; `"mtp": false` a retiré tout drafting (le prompt lookup `--spec`
  reste). La colonne sans drafting est la référence pour Magnetar.
- Le prompt de 16K n'a pas été mesuré : le contexte était de 8 192 tokens.
- Cellule du harness (prompt de la campagne, raisonnement actif, arrêt naturel
  vers 30 tokens) : 1,05 à 1,07 s par requête.

## Résultats et interprétation

Le résumé du job et l'artefact contiennent `comparison.md`, `status.tsv`, les
informations du matériel, les journaux et un `report.json` avec ses données
brutes par moteur. Un échec d'un moteur n'interrompt pas les suivants, mais
fait échouer le job ; les résultats partiels restent accessibles. Les modèles
sont résidents pendant la mesure, sauf les couches qu'AirLLM charge à la
demande. La latence et le débit de requête incluent le prompt et la génération.

Ces mesures demandent encore de vérifier leurs conditions avant de classer les
moteurs :

* Magnetar mesure ses requêtes en streaming dans sa campagne native ; AirLLM
  utilise une API Python synchrone ; llama.cpp et Ollama utilisent une API HTTP
  locale non streaming. Leurs frontières de mesure ne sont pas identiques.
* Magnetar et AirLLM utilisent les poids Hugging Face et des calculs F32 ;
  llama.cpp et Ollama chargent l'export GGUF FP16. Les précisions et les
  chemins de calcul ne sont donc pas identiques.
* Le cache de prompt de llama.cpp est désactivé. Ollama peut réutiliser son
  cache ; sa mesure peut donc comprendre moins de travail de préremplissage.
* Les tokens d'entrée et de sortie effectivement mesurés apparaissent dans le
  tableau. Le nombre de tokens peut différer ; les sorties et les données
  brutes doivent être vérifiées avant toute comparaison.
* Le placement CUDA est exigé pour Magnetar et AirLLM ; llama.cpp doit
  confirmer des couches déchargées et Ollama une occupation VRAM positive.
  Une valeur VRAM positive ne prouve pas que tout le modèle est sur GPU.
  Sur un CPU avec AMX, le backend CPU d'Ollama range les poids dans un
  tampon réarrangé que `/api/ps` compte dans `size_vram` : la cellule CPU
  l'accepte seulement si le journal du serveur ne signale que le CPU comme
  calcul découvert (`library=cpu`).
  Sur CPU, Magnetar utilise son Provider CPU optimisé (`cpu-native`), pas le Provider de référence.

Le scénario est court et mono-requête ; il ne permet pas de conclure sur des
contextes longs, la concurrence ou la performance générale. Le runner CPU
GitHub peut changer de processeur entre campagnes : comparer les moteurs au
sein d'un même job, puis répéter sur une machine stable pour une étude durable.

## Utilisation locale

La campagne CI Linux peut être lancée depuis la racine dans un environnement
adapté, avec Rust, CMake, git, curl, Python et les dépendances de construction :

```bash
export RUNNER_TEMP="$(mktemp -d)"
export BENCH_DEVICE=cpu BENCH_TOKENS=16 BENCH_RUNS=5
bash benchmark/campaign.sh
```

Pour CUDA, choisir `BENCH_DEVICE=gpu` et utiliser un environnement compatible
avec le toolkit, le pilote NVIDIA et PyTorch. La campagne télécharge plusieurs
gigaoctets et construit des composants ; prévoir suffisamment de disque.

Les adaptateurs historiques restent accessibles individuellement avec
`python benchmark/run.py --help`. Pour comparer leurs rapports individuels :
`python benchmark/compare.py chemin/vers/report1.json chemin/vers/report2.json`.
La suite de tests sans dépendance externe se lance avec
`python -m unittest discover -s benchmark/tests -v`.
