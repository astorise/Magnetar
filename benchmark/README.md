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
  Sur CPU, Magnetar utilise son provider CPU de référence.

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
