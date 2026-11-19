# Benchmarks d'inférence

Cette suite pilote cinq sous-projets : `magnetar`, `candle`, `llama_cpp`,
`ollama` et `strata`. Chaque appel mesure un moteur et un appareil. Elle
n'installe aucun moteur, ne télécharge aucun modèle et nécessite Python 3.9+
sans dépendance externe. Les résultats et les sorties brutes sont conservés
dans `benchmark/results/` (ignoré par Git).

Le scénario de base est une requête à la fois, le texte brut
`The capital of France is`, une génération gloutonne, 16 tokens au maximum,
un échauffement et cinq répétitions mesurées. Les moteurs peuvent s'arrêter
avant la limite ; les comptes réellement retournés sont conservés. Les
campagnes Magnetar exigent une limite d'au moins deux tokens afin de mesurer
la latence de décodage entre les tokens. Les adaptateurs utilisent les
interfaces réellement disponibles : les unités,
frontières de mesure et conditions d'exécution ne sont pas toutes identiques.
Il ne faut pas classer les moteurs à partir de valeurs de noms différents.

## Préparer les modèles et exécutables

1. Pour Magnetar et Candle, placer dans un dossier le checkpoint
   [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775)
   à la révision `7ae557604adf67be50417f59c2c2f167def9a775` :
   `config.json`, `generation_config.json`, `tokenizer.json`,
   `tokenizer_config.json` et `model.safetensors`. Magnetar vérifie le SHA-256
   du fichier de poids (`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`).
   Initialiser les sous-modules et installer la toolchain Rust du dépôt.
2. Construire l'exemple `qwen` de Candle en release depuis un checkout de
   [Candle](https://github.com/huggingface/candle). Par exemple, sous PowerShell,
   depuis la racine de ce dépôt :

   ```powershell
   cargo build --release --manifest-path ..\candle\Cargo.toml -p candle-examples --example qwen
   # Pour CUDA, ajouter : --features cuda
   ```

   Indiquer le binaire obtenu, par exemple
   `..\candle\target\release\examples\qwen.exe`. L'exemple Candle doit prendre
   en charge les options `--weight-path`, `--tokenizer-file` et
   `--no-chat-template`. L'adaptateur s'appuie sur un seul fichier de poids.
3. Construire [`llama-completion`](https://github.com/ggml-org/llama.cpp/tree/master/tools/completion)
   avec le backend voulu, et fournir un GGUF issu du même checkpoint. Relever
   la révision de conversion et le type des poids ; un GGUF quantifié n'est
   pas équivalent aux poids non quantifiés de Magnetar/Candle.
4. Démarrer [Ollama](https://github.com/ollama/ollama) séparément et y
   installer un modèle. Pour réutiliser un GGUF local, créer un Modelfile
   contenant `FROM <chemin-absolu-vers-le-fichier.gguf>`, puis exécuter
   `ollama create magnetar-bench -f <chemin-du-Modelfile>`.
5. Démarrer [Strata](https://github.com/Niko1221/Strata) séparément et
   configurer son modèle et son placement CPU/GPU. Spécifier sa racine HTTP
   et son nom de modèle. Si nécessaire, placer le jeton dans la variable
   `STRATA_API_KEY` ; il n'est pas écrit dans les rapports.

Strata vise notamment Qwen3.8-Flash-Next : il n'est pas établi que ce modèle
soit pris en charge par la campagne Magnetar. Si aucun checkpoint commun n'est
pris en charge par les cinq moteurs, les résultats Strata décrivent une charge
différente et ne constituent pas une comparaison directe.

## Lancer les cellules

Exemples PowerShell depuis la racine du dépôt, avec des chemins à adapter :

```powershell
$bundle = 'C:\models\Qwen2.5-0.5B-Instruct'
$gguf = 'C:\models\Qwen2.5-0.5B-Instruct-f16.gguf'
$qwen = 'C:\tools\candle\target\release\examples\qwen.exe'
$llama = 'C:\tools\llama.cpp\build\bin\Release\llama-completion.exe'

python benchmark/run.py magnetar --device cpu --model-path $bundle
python benchmark/run.py magnetar --device gpu --model-path $bundle
python benchmark/run.py candle --device cpu --model-path $bundle --binary $qwen
python benchmark/run.py candle --device gpu --model-path $bundle --binary $qwen
python benchmark/run.py llama_cpp --device cpu --model-path $gguf --binary $llama
python benchmark/run.py llama_cpp --device gpu --model-path $gguf --binary $llama
python benchmark/run.py ollama --device cpu --model magnetar-bench
python benchmark/run.py ollama --device gpu --model magnetar-bench
python benchmark/run.py strata --device cpu --url http://127.0.0.1:8096 --model <nom-du-modele>
python benchmark/run.py strata --device gpu --url http://127.0.0.1:8096 --model <nom-du-modele>
```

Remplacer `<nom-du-modele>` par le nom réellement exposé, sans les chevrons.
Relancer Strata avec la configuration CPU ou GPU appropriée entre les deux
commandes : `--device` décrit le serveur, il ne le reconfigure pas.
`--tokens 32`, `--runs 10`, `--output-dir`, `--revision` et `--notes` permettent
de modifier les paramètres et d'enregistrer version, pilote et configuration.
Garder les mêmes paramètres et une machine au repos pour toute campagne.

Chaque dossier de résultat contient `report.json` et `raw/` avec les données
originales. Une sortie existante n'est jamais écrasée. Afficher plusieurs
rapports avec :

```powershell
python benchmark/compare.py chemin/vers/report1.json chemin/vers/report2.json
python -m unittest discover -s benchmark/tests -v
```

## GitHub Actions

Les workflows **Benchmark CPU** et **Benchmark GPU** sont lancés manuellement
depuis l'onglet Actions. Chacun exécute le moteur sélectionné pour un seul
appareil : relancer le workflow pour chaque moteur à mesurer. Le CPU utilise
`ubuntu-latest` ; le GPU utilise `arc-gpu-magnetar` et la même image CUDA que
le test GPU du dépôt. Le workflow GPU partage le groupe de concurrence du
test GPU existant pour éviter leur exécution simultanée. Les résultats et les
informations sur le matériel sont téléchargeables comme artefact du job,
y compris les données partielles disponibles si le benchmark échoue.

Les paramètres `engine`, `tokens` et `runs` sélectionnent le moteur et la
charge ; `magnetar`, 16 tokens et cinq répétitions sont les valeurs par défaut.
Les fichiers Hugging Face et GGUF sont téléchargés à des révisions précises et
leurs SHA-256 sont vérifiés. Candle et llama.cpp sont construits depuis leurs
sources pour le backend choisi ; `engine_ref` permet de choisir un commit ou
une branche présente dans le clone. Par défaut, les commits
`cfcbd7186d13271843ce7eb41d25659bd0835850` pour Candle et
`bafa0420c1e57edf04d71587d32fe57d45df7b95` pour llama.cpp sont utilisés ;
le commit réellement construit figure également dans le rapport.

Pour Ollama et Strata, renseigner `server_url` et `server_model` et démarrer le
serveur avant le lancement. Le serveur doit être accessible **depuis le job**
(le GPU tourne dans un conteneur, dont `localhost` n'est pas celui de l'hôte).
Configurer le modèle et le placement avant la mesure. Si Strata exige une clé,
la définir dans le secret GitHub Actions `STRATA_API_KEY`. Ces services ne sont
pas démarrés par le workflow et peuvent tourner sur une autre machine : dans ce
cas le CPU/GPU du serveur est celui qui est mesuré. Les informations du runner
ne prouvent pas le placement de Strata. Pour Ollama, le contrôle de placement
du modèle est décrit ci-dessous.

## Interprétation

| Sous-projet | Mesure et limites |
| --- | --- |
| Magnetar | Campagne native de qualification : modèle résident, requête froide, échauffement puis requêtes mesurées en streaming. TTFT, débit et traces par token ; Provider CPU optimisé (`cpu-native`) ou CUDA exigé. Le rapport natif contient aussi les ressources et les limites de qualification. |
| Candle | Exemple Qwen : nouveau processus et rechargement à chaque essai. `generation_tokens_per_second` inclut le traitement du prompt ; `process_seconds_including_load` inclut le démarrage et le chargement. CPU F32, GPU BF16 selon l'exemple. Le repli CPU explicite est refusé en mode GPU. |
| llama.cpp | `llama-completion`, nouveau processus et rechargement à chaque essai. Débit de décodage lu dans les timings natifs. `--n-gpu-layers 0` en CPU, `999` demandé en GPU : vérifier dans les journaux les couches réellement déchargées et le backend utilisé. |
| Ollama | API `/api/generate` sans template ; durée de requête et débit de décodage natif. Le placement est vérifié par `size_vram` dans `/api/ps`. Une valeur positive peut représenter un placement partiel GPU/CPU. Redémarrer ou décharger le modèle entre les modes CPU et GPU si nécessaire. |
| Strata | API `/v1/chat/completions` non streaming : le serveur applique son template et son cache peut réutiliser des préfixes. Le débit est `completion_tokens / durée HTTP`, pas un débit de décodage. Le placement matériel et l'identité exacte des poids restent à vérifier côté serveur. |

Pour une comparaison fiable, vérifier les hachages, la conversion, la
quantification, le nombre de tokens d'entrée et de sortie, les templates, la
précision de calcul, le placement réel, les options et la version de chaque
moteur. Les processus frais, modèles résidents et requêtes HTTP ont des coûts
différents. Les métriques indisponibles ne sont ni estimées ni inventées.
