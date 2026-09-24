# Audit complet — Magnetar

**Date :** 24 septembre 2026  
**Révision :** mise à jour après décision de publication des artefacts Magnetar  
**Dépôt :** `astorise/Magnetar`  
**Branche auditée :** `main`  
**SHA courant :** `78e8eb9b26d54a05d971e0e2f4c3f0163b869e49`  
**SHA embarqué par Tachyon :** `9db481de9dd43efcd54d06b62f0b90d6cc61aca3`  
**Écart entre les deux :** 1 commit, exclusivement documentaire (`README.md` + rapport d’audit)  
**Périmètre :** architecture, contrat Tachyon, sécurité, trust, chargement d’artefacts, concurrence, qualité, tests, CI/CD, dépendances, supply chain, documentation et gouvernance.

## 1. Verdict

### Verdict architecture / intégration Tachyon : 🟢 GO

La séparation recherchée est maintenant effective :

```text
Tachyon
  ├── transport de Component
  ├── provenance / trust configuré par l’hôte
  ├── routage / QoS / admission
  ├── placement générique
  ├── deadline / annulation
  └── payload opaque
        │
        ▼
Magnetar
  ├── Component WASM
  ├── Model Artifact
  ├── format d’artefact
  ├── tokenizer / chat template
  ├── architecture modèle
  ├── ModelInstance
  ├── Provider / Device
  ├── planification / kernels
  ├── KV cache
  ├── génération
  └── streaming
```

Les blockers relevés lors des audits précédents sont fermés : le registre générique n’est plus Qwen-shaped, le guard d’isolation couvre le registre Component, une vraie seconde architecture Llama est exercée, le format d’artefact est explicite, le trust Component et le trust Model Artifact sont distincts, et un cache hit ne contourne pas la politique de trust.

### Verdict qualité du code : 🟢 GO

Le workflow `Quality` du SHA courant `78e8eb9…` est vert sur les gates structurantes : Rustfmt, Clippy, Windows/macOS/Linux, MSRV, docs, WIT, OpenSpec, couverture, cargo-deny, Wasmtime Component Engine, Provider conformance, Component integration, format integration, E2E conformance et isolation des familles de modèles.

### Verdict production / gouvernance : 🟠 GO CONDITIONNEL

Il reste un point de gouvernance bloquant pour une clôture « production » stricte : la branche `main` apparaît **non protégée** côté GitHub (`protected: false`) et aucun ruleset de dépôt n’est visible. Les contrôles CI sont excellents, mais ils ne constituent pas une garantie s’ils peuvent être contournés par un push direct.

Deux limites connues restent également à assumer explicitement : la signature cryptographique des artefacts n’est pas implémentée, et une instance `LoadedInferenceComponent` ne traite qu’une génération active à la fois.

Depuis cette révision, **la publication des artefacts est une exigence de livraison décidée**. Magnetar doit désormais disposer d’une chaîne de publication reproductible et vérifiable : `magnetar-runtime` sur crates.io, binaires/rapports/SBOM/provenance via GitHub Releases, et Components / Kernel Exchange Bundles via un registre OCI. Tant que cette chaîne n’est pas effectivement implémentée et vérifiée, Magnetar reste architecturalement stable mais ne doit pas être présenté comme disposant d’un canal de distribution officiel complet.

## 2. Synthèse des findings

| ID | Sévérité | État | Finding |
|---|---:|:---:|---|
| MAG-01 | P1 | Ouvert | `main` n’est pas protégée ; les gates CI/CODEOWNERS peuvent donc être contournés au niveau gouvernance. |
| MAG-02 | P2 | Planifié | Les signatures cryptographiques de Component/Model Artifacts et l’identité éditeur authentifiée restent « design only » ; leur implémentation est désormais couplée au chantier de publication. |
| MAG-03 | P2 | À traiter dans l’intégration | Une `LoadedInferenceComponent` sérialise les générations ; aucun scheduler/batching/instance pool n’est fourni par cette façade. |
| MAG-04 | P3 | Durcissement | La façade `inference-components` expose encore publiquement `ComponentProviderAdvertisement` / `ProviderDeviceClass`, même si Tachyon ne les consomme plus. |
| MAG-05 | P3 | Accepté | Le `cargo deny` du workspace submodules autorise explicitement certains wildcards / métadonnées de licence absentes provenant de sous-modules externes. |
| MAG-06 | P1 | Ouvert — décidé | La chaîne officielle de publication/distribution n’est pas encore implémentée : crates.io pour `magnetar-runtime`, GitHub Releases pour les binaires et preuves de release, OCI pour Components/Kernels. |

**P0 identifié : aucun.**

## 3. Architecture

### 3.1 Runtime générique

Le runtime principal est désormais réellement structuré par contrats :

- `ProductionModelArtifactIngestor` est une abstraction générique ;
- `ProductionIngestionRegistry` stocke des `Arc<dyn ProductionModelArtifactIngestor>` plutôt que des types concrets de loaders ;
- les loaders Hugging Face / GGUF restent derrière la couche de composition `inference-components` ;
- le runtime ne déduit plus le format d’un modèle en inspectant opportunistement le filesystem ;
- le Component choisi par l’appelant reste l’autorité explicite de construction/exécution du graphe.

**État : conforme.**

Sources :
- `magnetar-runtime/src/production_model_ingestion.rs`
- `magnetar-runtime/src/component.rs`
- `inference-components/src/lib.rs`
- `tools/check_generic_facade_family_isolation.py`

### 3.2 Format d’artefact explicite

Le fichier :

```text
magnetar-artifact-format.yaml
```

est la source normative du format (`huggingface`, `gguf`, etc.).

`read_declared_artifact_format()` échoue explicitement en cas de sidecar absent, malformé ou de valeur non reconnue. L’ancien mécanisme de détection par structure de fichiers n’est conservé que dans une migration legacy explicitement appelée.

C’est la bonne direction : **pas de fallback implicite, pas d’ambiguïté silencieuse**.

**État : conforme.**

### 3.3 Deuxième architecture réelle

Les correctifs des audits précédents ont introduit une vraie voie Llama distincte, avec Component, manifeste, config, famille d’architecture et tensors propres. L’incompatibilité d’un Component Qwen avec un Artifact Llama est rejetée.

Cela ferme correctement le risque d’une abstraction « générique » uniquement en apparence.

**État : conforme.**

## 4. Sécurité et trust

### 4.1 Séparation Component / Model Artifact

`ArtifactTrustPolicy` maintient deux décisions indépendantes :

```text
Component WASM          -> ComponentTrustStore
Model Artifact          -> ModelTrustStore
```

Le fait de faire confiance aux poids ne fait pas confiance au code WASM, et inversement.

Le digest du Component est calculé depuis les bytes réels et comparé à la politique fournie par l’appelant. La politique de Model Artifact est appliquée séparément.

**État : conforme.**

### 4.2 Cache et trust

Le design empêche un artefact présent dans le cache de devenir implicitement trusted. La politique du caller est réévaluée.

**État : conforme.**

### 4.3 Source filesystem autorisée

`ProductionModelSource::resolve()` :

- refuse les chemins absolus ;
- refuse `..` et les composants non normaux ;
- canonicalise la racine ;
- canonicalise le candidat ;
- rejette les sorties de racine, y compris via symlink.

C’est une bonne barrière contre traversal et symlink escape.

**État : conforme.**

### 4.4 MAG-02 — signatures cryptographiques non implémentées

`SECURITY.md` indique explicitement que :

- les signatures Component / Model Artifact ne transportent pas encore de preuve cryptographique vérifiée ;
- l’identité publisher/source reste de la métadonnée ;
- la confiance en production dépend aujourd’hui du pinning explicite de digest, ou de la politique locale de développement.

`docs/cryptographic-artifact-signatures.md` est marqué :

> `Status: design only, not implemented.`

Le design Ed25519 est sérieux et fail-closed, mais l’issue de design #37 est fermée sans qu’un chantier d’implémentation ouvert soit visible.

### Risque

Ce n’est **pas** un bypass de la politique actuelle : le système échoue correctement si aucun digest trusted ne correspond.

En revanche, Magnetar ne peut pas encore établir cryptographiquement :

```text
"cet artefact a été publié par l’éditeur X"
```

sans pré-pinner chaque nouveau digest.

### Recommandation

Ouvrir un chantier d’implémentation séparé et conserver jusque-là une formulation stricte dans la documentation :

> Trust de production = digest pinning explicite ; publisher/source metadata non authentifiée.

**Sévérité : P2**, tant que le digest pinning reste le mécanisme de confiance officiellement supporté.

## 5. Concurrence et performances

### 5.1 Modèle actuel

`LoadedInferenceComponent` documente clairement :

> **One active generation at a time per instance.**

Le `ProductionLoadedModel` est protégé par un `Mutex` pendant toute la génération. Une deuxième requête :

- n’est pas rejetée ;
- n’est pas intercalée ;
- attend la fin de la précédente.

La façade ne fournit elle-même :

- ni scheduler ;
- ni continuous batching ;
- ni pool d’instances ;
- ni réplication automatique.

L’embedder doit charger plusieurs instances et router entre elles.

### Impact

Ce comportement est correct fonctionnellement et cohérent avec l’état mutable du KV cache. Il devient toutefois un point majeur dès que Magnetar est intégré dans un serveur à concurrence élevée.

**Ce n’est pas un defect de concurrence caché : c’est une limite volontaire et documentée.**

La responsabilité d’ajouter le pool est donc surtout du côté de l’embedder — aujourd’hui Tachyon.

## 6. Façade d’intégration

`inference-components` expose encore :

```rust
ProviderDeviceClass
ComponentProviderAdvertisement
LoadedInferenceComponent::provider()
```

Tachyon ne consomme plus ces types depuis le correctif round 5, et son guard interdit de les réintroduire dans son core.

### Risque

La surface publique garde malgré tout un chemin tentant pour un futur consommateur qui voudrait refléter l’identité interne du Provider au lieu de rester sur un placement générique.

### Recommandation

À moyen terme :

- rendre ces éléments internes si aucun consumer externe légitime n’en a besoin ;
- ou exposer une capability opaque/stable distincte de l’identité Provider.

**Sévérité : P3.**

## 7. Qualité et tests

Le SHA courant dispose d’une couverture CI exceptionnellement large pour un projet pré-1.0.

### Gates observées vertes

- `rustfmt`
- `clippy`
- Linux / Windows / macOS
- MSRV
- documentation Rust avec warnings denied
- WIT
- OpenSpec strict
- coverage ratchet
- `cargo deny`
- Wasmtime Component Engine
- wasm32 Component Engine
- model-family isolation
- Provider conformance
- Provider integration
- Component integration
- format integration
- E2E conformance
- submodule integration
- test-code placement

Workflow :
- `https://github.com/astorise/Magnetar/actions/runs/35911046736`

**État : très bon.**

## 8. Supply chain

### 8.1 GitHub Actions

Toutes les occurrences `uses:` inspectées dans les workflows Magnetar sont épinglées sur des **SHA Git complets**, par exemple :

```text
actions/checkout@3d3c42e5...
Swatinem/rust-cache@6323deb1...
taiki-e/install-action@6a0cb247...
```

Aucun `pull_request_target` n’a été trouvé.

**État : conforme et supérieur au niveau de Tachyon sur ce point.**

### 8.2 Rust

- Rust épinglé : `1.98.1`
- `Cargo.lock` versionné
- `cargo deny` en CI
- sources inconnues interdites
- wildcards interdits dans le workspace principal
- licences permissives explicites

Une exception RUSTSEC documentée existe pour `paste` (`RUSTSEC-2024-0436`, unmaintained, sans correctif disponible via la dépendance `tokenizers`).

**État : acceptable avec dette connue documentée.**

### 8.3 Décision de publication des artefacts — MAG-06

La stabilité atteinte par Magnetar permet maintenant de figer un **contrat de distribution officiel**.

La décision retenue est de ne pas inventer un registre propriétaire Magnetar, mais de séparer les canaux selon la nature de l’artefact :

| Artefact | Canal officiel cible | Identité normative |
|---|---|---|
| `magnetar-runtime` | crates.io | SemVer + crate checksum |
| CLI et binaires de release | GitHub Releases | tag Git + checksum |
| rapports de conformance, SBOM, provenance, checksums | GitHub Releases | tag Git + digest |
| Component Artifacts WASM | registre OCI, GHCR comme premier backend | digest OCI `sha256:` |
| Kernel Exchange Bundles | registre OCI | digest OCI `sha256:` |
| petits fixtures/modèles de conformance | OCI ou release selon usage | digest |
| Model Artifacts volumineux | source externe autorisée (HF/S3/OCI/Tachyon, etc.) | digest de contenu + manifeste Magnetar |

#### Principes d’architecture

1. **Le Runtime reste transport-neutral.** La résolution réseau, les credentials et le téléchargement ne deviennent pas des autorités implicites du Runtime Core.
2. **Le tag n’est jamais l’identité de confiance.** Un tag (`0.2.0`, `latest`, alias humain) ne sert qu’à la résolution. L’identité de contenu reste le digest.
3. **Le canal de distribution ne confère pas la confiance.** Un artefact provenant de GHCR, d’un registre privé, de Tachyon ou d’un stockage objet passe par le même ingestion/trust gateway.
4. **La publication doit produire les preuves de release.** Checksums, SBOM, provenance, rapports de conformance et métadonnées de build doivent être générés depuis le commit/tag final.
5. **Les signatures cryptographiques sont un chantier associé, pas un substitut au digest.** Le digest pinning reste valide et fail-closed. La signature/publisher identity permettra ensuite une politique de confiance dynamique.

#### Types OCI Magnetar à formaliser

Le chantier devra définir des media types versionnés, par exemple :

```text
application/vnd.magnetar.component.v1+wasm
application/vnd.magnetar.component-manifest.v1+json
application/vnd.magnetar.kernel-bundle.v1+tar+zstd
application/vnd.magnetar.kernel-manifest.v1+json
application/vnd.magnetar.conformance.v1+json
```

Les noms exacts sont à figer dans OpenSpec avant publication stable.

#### Ordre de mise en œuvre recommandé

```text
release tag
   │
   ├── cargo publish magnetar-runtime
   │
   ├── GitHub Release
   │     ├── CLI / binaires
   │     ├── checksums
   │     ├── SBOM
   │     ├── provenance
   │     └── rapports de conformance
   │
   └── OCI publish
         ├── Components
         ├── Kernel bundles
         └── fixtures de conformance
```

Puis, côté consommation :

```text
resolve alias/tag
      ↓
resolve immutable digest
      ↓
download bytes
      ↓
verify digest
      ↓
verify trust/signature policy
      ↓
local content-addressed cache
      ↓
Magnetar ingestion gateway
      ↓
Runtime
```

#### Portée Model Artifact

Les poids de modèles volumineux ne doivent pas être recopiés systématiquement dans un registre Magnetar. Magnetar publie et consomme un **manifeste d’identité et d’intégrité**, tandis que les octets peuvent rester sur une source autorisée externe.

Cette décision préserve la séparation :

```text
distribution / credentials / transport
                ≠
identity / integrity / trust
```

#### Critère de fermeture de MAG-06

MAG-06 pourra être fermé lorsque :

- `cargo publish --dry-run` puis la publication réelle de `magnetar-runtime` sont validés ;
- une release GitHub issue d’un tag protégé contient binaires + preuves de release ;
- au moins un Component Artifact est publié en OCI et réingéré par digest ;
- au moins un Kernel Exchange Bundle est publié/récupéré par le même contrat de distribution ;
- un test prouve qu’un changement de tag ne contourne pas le digest/trust ;
- les checksums/SBOM/provenance correspondent aux octets réellement publiés ;
- la procédure de retrait/correction d’une release invalide est documentée.

**Sévérité : P1 pour la capacité de publication officielle**, sans remettre en cause le GO architectural du Runtime.

### 8.4 Sous-modules

Le workspace des sous-modules exécute :

```text
cargo deny ... --allow wildcard --allow unlicensed
```

car certains dépôts externes ne publient pas toute la métadonnée attendue.

Ce n’est pas un blocker puisque les sous-modules sont contrôlés ailleurs dans la CI, mais cette différence doit rester visible.

**Sévérité : P3.**

## 9. Gouvernance

### MAG-01 — branche principale non protégée

L’API GitHub renvoie actuellement :

```text
main.protected = false
```

et le dépôt ne publie aucun repository ruleset visible via l’API.

`CODEOWNERS` existe et protège conceptuellement les zones sensibles :

- architecture / OpenSpec ;
- workflows ;
- release / coverage ;
- sécurité ;
- submodules.

Mais **CODEOWNERS sans règle de branche exigeant la review n’est pas une barrière**.

### Risque

Un push direct sur `main` peut théoriquement contourner :

- review ;
- checks requis ;
- contrôle CODEOWNERS ;
- politique de merge.

Pour un runtime qui charge du code WASM et des Providers natifs, ce point doit être traité comme une exigence de release.

### Recommandation minimale

Protéger `main` ou créer un ruleset avec :

- PR obligatoire ;
- au moins 1 review ;
- Code Owner review sur zones sensibles ;
- checks `Quality` requis ;
- branche à jour avant merge ;
- blocage des force-push ;
- blocage de suppression de branche ;
- idéalement conversation resolution.

**Sévérité : P1.**

## 10. Documentation

La documentation de l’architecture actuelle est nettement meilleure que lors des premiers audits :

- responsabilités explicites ;
- security model documenté ;
- limites de concurrence documentées ;
- format d’artefact explicite ;
- trust explicite ;
- release security clairement séparée des promesses non implémentées.

Le dernier commit `78e8eb9…` ne modifie que :

- `README.md`
- `docs/audits/audit-magnetar-integration-tachyon-2026-09-23-closure.md`

Il n’y a donc **aucune divergence de code** avec le SHA `9db481d…` vendored par Tachyon.

## 11. Plan de clôture recommandé

### Bloquant avant clôture production / publication officielle

1. **MAG-01** — protéger `main` et rendre les gates CI obligatoires.
2. **MAG-06** — implémenter et vérifier la chaîne officielle de publication :
   - `magnetar-runtime` → crates.io ;
   - CLI + rapports + SBOM + provenance + checksums → GitHub Releases ;
   - Components + Kernel Exchange Bundles → OCI/GHCR, adressés par digest.

### À programmer dans le même mouvement ou immédiatement après

3. **MAG-02** — implémenter la signature cryptographique / publisher identity authentifiée. La première publication peut rester sûre par digest pinning explicite, mais les signatures doivent devenir la voie de confiance dynamique.
4. Réduire la surface Provider publique de `inference-components` si elle n’a pas de consumer légitime.
5. Conserver le modèle de concurrence actuel comme primitive ou fournir ultérieurement une factory/pool — sans prétendre qu’une instance offre du parallel serving.

### Séquence de validation avant ré-audit Tachyon

Une fois MAG-01/MAG-06 traités, le prochain audit Magnetar devra vérifier les **octets réellement publiés**, et non uniquement la configuration du pipeline :

```text
commit protégé
 → tag
 → build reproductible
 → publication
 → digest/checksum
 → SBOM/provenance
 → téléchargement propre
 → réingestion Magnetar
 → conformance
```

Ce n’est qu’après cette validation qu’il sera pertinent de mettre à jour l’audit Tachyon sur son rôle de consumer/distributeur des artefacts Magnetar.

## 12. Critères de sortie

| Critère | État |
|---|:---:|
| Séparation Tachyon / Magnetar | ✅ |
| Component externe explicite | ✅ |
| Pas de default Component implicite | ✅ |
| Trust Component séparé | ✅ |
| Trust Model Artifact séparé | ✅ |
| Cache hit ne bypass pas le trust | ✅ |
| Format Artifact explicite | ✅ |
| Path traversal / symlink escape bloqués | ✅ |
| Architecture-neutral runtime | ✅ |
| Vraie seconde architecture | ✅ |
| Incompatibilité architecture rejetée | ✅ |
| Actions CI pinées par SHA | ✅ |
| Contrat de publication officiel décidé | ✅ |
| Publication crates.io de `magnetar-runtime` | ⏳ à implémenter |
| Publication GitHub Release + preuves | ⏳ à implémenter |
| Publication OCI Component Artifact | ⏳ à implémenter |
| Publication OCI Kernel Exchange Bundle | ⏳ à implémenter |
| Réingestion d’un artefact publié par digest | ⏳ à vérifier |
| Toolchain Rust pinée | ✅ |
| Cargo lock versionné | ✅ |
| Cargo deny vert | ✅ |
| Quality complète verte | ✅ |
| SHA vendored Tachyon code-identique au `main` courant | ✅ |
| Signature cryptographique publisher | ⚠️ non implémentée |
| Parallel serving par instance | ⚠️ non supporté |
| `main` protégée / checks obligatoires | ❌ |

## 13. Conclusion

**Magnetar a fermé les problèmes d’architecture et d’intégration qui justifiaient les précédents audits.**

Le code actuel est cohérent, fortement testé et bien gardé contre les régressions de frontière. Sur le contrat Magnetar ↔ Tachyon, le projet est **GO**.

Je ne qualifierais cependant pas le dépôt de « totalement clôturé pour une release de production » tant que la branche principale reste non protégée **et que la nouvelle chaîne officielle de publication n’a pas été mise en œuvre puis vérifiée sur les artefacts réellement publiés**.

La décision de publication ne remet pas en cause le GO architectural : elle transforme désormais la distribution en exigence explicite de produit. Le Runtime doit rester transport-neutral et content-addressed ; crates.io, GitHub Releases et OCI sont des canaux de publication, jamais des autorités de confiance implicites.

Les signatures cryptographiques et le modèle une-génération-par-instance restent des limites connues. La signature est désormais naturellement rattachée au chantier de publication ; le modèle de concurrence reste principalement une responsabilité d’intégration/serving.

**Prochaine étape d’audit prévue :** une fois la publication Magnetar effectivement réalisée, vérifier la release de bout en bout (tag → artefacts → digests → preuves → réingestion), puis seulement mettre à jour l’audit Tachyon sur la consommation/distribution de ces artefacts.

### Sources principales

- `https://github.com/astorise/Magnetar/tree/78e8eb9b26d54a05d971e0e2f4c3f0163b869e49`
- `https://github.com/astorise/Magnetar/blob/78e8eb9b26d54a05d971e0e2f4c3f0163b869e49/SECURITY.md`
- `https://github.com/astorise/Magnetar/blob/78e8eb9b26d54a05d971e0e2f4c3f0163b869e49/inference-components/src/lib.rs`
- `https://github.com/astorise/Magnetar/blob/78e8eb9b26d54a05d971e0e2f4c3f0163b869e49/magnetar-runtime/src/production_model_ingestion.rs`
- `https://github.com/astorise/Magnetar/blob/78e8eb9b26d54a05d971e0e2f4c3f0163b869e49/tools/check_generic_facade_family_isolation.py`
- `https://github.com/astorise/Magnetar/blob/78e8eb9b26d54a05d971e0e2f4c3f0163b869e49/docs/cryptographic-artifact-signatures.md`
- `https://github.com/astorise/Magnetar/blob/78e8eb9b26d54a05d971e0e2f4c3f0163b869e49/docs/release-security.md`
- `https://github.com/astorise/Magnetar/actions/runs/35911046736`
