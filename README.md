# VisionPack: DatasetOps para Visão Computacional

## Project Status

VisionPack is in an early MVP stage. The current codebase already provides an installable Python package and a CLI named `vp`.

For practical usage, see [docs/usage.md](docs/usage.md).

### Done

```bash
vp init --name factory-defects --task detection
vp import ./raw --format yolo
vp import ./coco.json --format coco --images ./images
vp validate --strict
vp stats
vp stats --by split
vp split create --train 0.8 --val 0.1 --test 0.1 --strategy stratified
vp split lock
vp snapshot create -m "initial import"
vp snapshot list
vp diff v1 v2
vp export --format yolo --output exports/yolo-v1
vp export --format coco --output exports/coco-v1
vp pack --profile archive
```

Implemented:

- Python package structure split across CLI, core, storage, index, formats, validation, packing, and tests
- `uv` dependency management with `uv.lock`
- `visionpack.yaml` manifest creation and `pydantic`-validated parsing with actionable schema errors
- local content-addressed asset store under `.vp/objects/sha256`
- JSON local index under `.vp/db/index.json` with cached, O(1) annotation lookups (DuckDB-ready interface)
- robust image probing via Pillow (webp/EXIF-orientation correct), reading each file once for hash + probe + store
- YOLO detection import (parallelized) with image hashing, class discovery, normalized label parsing, and internal bounding-box conversion
- COCO detection import and export (instances JSON)
- multi-source merging: classes from different sources merge by name (YOLO labels mapped via the source's own class order, not positionally)
- deterministic, versionable train/val/test splits (stratified / random / hash strategies, seeded, lockable, captured in snapshots)
- per-split statistics for comparable metrics as the dataset grows
- validation for unreadable images, missing annotations, orphan labels, unknown classes, invalid bounding boxes, duplicate content, and split leakage
- dataset statistics
- content-addressed local snapshots (deduplicated inventory blobs) with manifest, asset, annotation, split, and stats hashes
- diff between snapshots
- YOLO and COCO export
- archive packing to `.tar.zst`
- integration tests for the YOLO and COCO flows, plus media, manifest, and snapshot units

### Roadmap

The roadmap is sequenced so that each phase unblocks the next. Phases A and C
make the tool correct and scalable; phase B contains the differentiators that
make VisionPack worth adopting over a pile of scripts.

#### Phase 0 — Foundations (done)

Make the core correct and able to handle the README's own target scale
(~12k images / 47k annotations) before adding features on top.

- [x] index access without O(n²) scans (cache deserialized records; build
      `asset_id -> annotation` lookup once) — `index/json_index.py`
- [x] parallelize import hashing + image probing, reading each file once for
      hash + probe + store (`formats/yolo.py`, `media.py`, `storage/`)
- [x] fix `.webp` import crash and EXIF-orientation width/height swaps by probing
      images with Pillow (`media.py`), with regression tests
- [x] validate manifest + `visionpack.yaml` with `pydantic` for actionable,
      field-level schema errors (`core/manifest.py`), with tests

#### Phase A — Scale & format coverage

- [x] content-address snapshots: store the inventory as a deduplicated blob under
      `.vp/snapshots/blobs/<hash>.json`; `vN.json` references `inventory_hash` and
      `load_snapshot` rehydrates it transparently (`snapshot.py`), packed in archives
- [x] implement COCO import (`vp import coco.json --format coco --images ./images`)
- [x] implement COCO export (`vp export --format coco`)
- [x] multi-source class merging: YOLO labels map via the source's own class names
      and unseen classes merge into the manifest by name (no more positional
      mislabeling); COCO categories merge on re-import (`core/manifest.py`, `formats/`)
- [x] deterministic, versionable splits (`vp split create/lock/list/show`):
      content-hash + seed assignment; `stratified` (default), `random` (exact ratios),
      `hash` (stable as data grows); lock freezes a split and snapshots capture it (`split.py`)
- [x] per-split stats for comparable metrics (`vp stats --by split`) + deterministic ordering
- [ ] implement `vp pack --profile training` (WebDataset shards)
- [ ] DuckDB index — **deferred by decision**: at current scale the cached lists are
      fine, and DuckDB's payoff (SQL aggregations/joins, scale past RAM) is pulled by
      Phase B. When built, use an in-memory DuckDB query layer over the portable JSON
      store (no binary index file, no file-lock issues), driven by a real Phase B query.

#### Phase B — Differentiators (what makes it essential)

- [ ] **near-duplicate & cross-split leakage detection**: fast perceptual hash
      (pHash/dHash) by default, optional embedding tier (CLIP/DINOv2) behind an extra;
      surface train↔test leakage and near-dup clusters in `vp validate`
- [ ] **label-health audit** (`vp audit`): high-IoU duplicate boxes, degenerate/edge-pinned
      boxes, per-class aspect-ratio outliers, class imbalance, scored report
- [ ] **model-in-the-loop quality** (optional extra, keeps core PyTorch-free): surface
      confident detections with no matching label, and confident disagreements with labels
- [ ] **distribution-drift diff**: class-distribution and resolution drift between snapshots
      (per-class deltas / KL divergence), not just added/removed IDs
- [ ] **dataset → model lineage**: `vp snapshot tag v4 trained:<run-id>` plus a metrics blob,
      to make the reproducibility claim real

#### Phase C — Reporting & polish

- [ ] add HTML validation, stats, and drift reports
- [ ] add JSON report output for stats and snapshot diff workflows
- [ ] add richer terminal output with `rich`
- [ ] move CLI plumbing from `argparse` to `typer` once command behavior stabilizes
- [ ] expand fixture coverage with malformed YOLO/COCO datasets

#### Later

- [ ] implement `vp annotate prepare` and `vp annotate ingest`
- [ ] add CVAT and Label Studio annotation package support
- [ ] add dataset-card generation
- [ ] add active-learning queue (rank unlabeled images by model uncertainty)
- [ ] add remote storage integrations (S3/GCS/Azure)
- [ ] add optional PyTorch dataset helpers

## 1. Objetivo

Projetar e implementar uma ferramenta open source chamada **VisionPack**, focada em organização, validação, versionamento, compressão e preparação de datasets para pipelines de visão computacional.

A ferramenta deve resolver problemas reais em times de computer vision:

- datasets desorganizados
- labels inconsistentes
- versões manuais e irreproduzíveis
- exports quebrados entre YOLO, COCO, CVAT e outros formatos
- splits contaminados entre treino, validação e teste
- compressão improvisada
- dificuldade de rastrear qual dataset treinou qual modelo
- dificuldade de preparar pacotes para anotação e revisão
- pipelines complexas baseadas em scripts soltos

O produto deve ser útil para pesquisadores, startups, times industriais de visão computacional e equipes que treinam modelos de detecção, segmentação, classificação e tracking.

A visão do produto é:

> Um Git/Docker-like para datasets de visão computacional: versionável, validável, comprimível, rastreável e pronto para treino ou anotação.

---

## 2. Princípios de Design

### 2.1 CLI-first

A primeira interface deve ser CLI. Não começar por web app.

A ferramenta precisa funcionar bem em:

- notebooks
- servidores de treino
- CI/CD
- máquinas locais
- pipelines com Makefile, Airflow, Prefect, Dagster ou GitHub Actions

### 2.2 Manifesto explícito

Todo dataset deve ter um arquivo declarativo central:

```text
visionpack.yaml
```

Esse arquivo descreve:

- nome do dataset
- tipo de tarefa
- classes
- formatos de entrada e saída
- splits
- políticas de validação
- perfis de compressão
- transformações
- integrações

### 2.3 Dados imutáveis, metadados versionáveis

Assets brutos, como imagens e vídeos, devem ser tratados como conteúdo imutável, identificados por hash.

Anotações, splits e transformações devem ser versionáveis.

A ferramenta não deve depender de “pasta com nome certo” como fonte da verdade. A fonte da verdade deve ser o manifesto + índice interno.

### 2.4 Interoperabilidade acima de lock-in

VisionPack não deve tentar substituir CVAT, FiftyOne, DVC, Label Studio, Roboflow, Datumaro ou lakeFS.

Ele deve atuar como camada de DatasetOps:

- importa de vários formatos
- valida
- organiza
- versiona
- comprime
- exporta
- prepara treino
- prepara pacotes de anotação
- rastreia linhagem

### 2.5 Reprodutibilidade

Deve ser possível responder:

- qual versão do dataset treinou determinado modelo?
- quais imagens entraram ou saíram entre duas versões?
- quais labels mudaram?
- qual split foi usado?
- quais transforms foram aplicadas?
- qual export gerou determinado artefato?
- houve vazamento entre treino e teste?

---

## 3. Escopo Inicial

O MVP deve suportar primeiro:

- imagens
- object detection
- formatos YOLO e COCO
- snapshots locais
- validação básica e intermediária
- compressão para treino e arquivamento
- splits versionados
- export para treino
- pacotes para anotação

Fora do MVP inicial:

- vídeo
- tracking
- segmentação avançada
- UI web completa
- storage remoto nativo
- colaboração multiusuário
- deduplicação semântica com embeddings
- integração profunda com Kubernetes

Esses itens podem entrar depois.

---

## 4. Sintaxe da CLI

A CLI deve ser clara, previsível e composável.

### 4.1 Inicialização

```bash
vp init
vp init --name factory-defects --task detection
```

Cria:

```text
visionpack.yaml
.vp/
assets/
annotations/
exports/
```

### 4.2 Importação

```bash
vp import ./raw --format yolo
vp import ./coco.json --format coco --images ./images
vp import ./dataset --format auto
```

Opções importantes:

```bash
vp import ./raw \
  --format yolo \
  --task detection \
  --copy hardlink \
  --class-map classes.yaml
```

Modos de cópia:

- `copy`: copia arquivos
- `move`: move arquivos
- `hardlink`: evita duplicação local
- `reference`: apenas referencia caminho externo
- `ingest`: copia para content-addressable store

### 4.3 Validação

```bash
vp validate
vp validate --strict
vp validate --fix
vp validate --report reports/validation.html
```

Validações iniciais:

- imagem corrompida
- label sem imagem
- imagem sem label
- classe desconhecida
- bounding box fora dos limites
- bounding box com área zero
- duplicatas exatas
- item presente em mais de um split
- classes ausentes em splits
- schema inválido
- resolução fora de limites configurados

### 4.4 Estatísticas

```bash
vp stats
vp stats --by class
vp stats --by split
vp stats --html reports/stats.html
```

Deve mostrar:

- número de imagens
- número de labels
- distribuição por classe
- distribuição por split
- resoluções
- tamanhos de arquivo
- imagens sem anotação
- labels por imagem
- outliers

### 4.5 Splits

```bash
vp split create --train 0.8 --val 0.1 --test 0.1
vp split create --strategy stratified --by class
vp split lock
vp split diff baseline current
```

Splits devem ser objetos versionáveis, não apenas pastas.

### 4.6 Snapshots

```bash
vp snapshot create -m "baseline inicial"
vp snapshot list
vp snapshot show v1
vp snapshot restore v1
```

Snapshot deve capturar:

- manifesto
- índice de assets
- índice de annotations
- splits
- transforms declaradas
- validações executadas
- estatísticas resumidas

### 4.7 Diff

```bash
vp diff v1 v2
vp diff v1 v2 --visual
vp diff v1 v2 --json
```

Deve responder:

- imagens adicionadas/removidas
- labels adicionadas/removidas/modificadas
- classes adicionadas/removidas
- mudanças nos splits
- mudanças de distribuição
- assets duplicados
- possíveis regressões

### 4.8 Compressão

```bash
vp pack --profile training
vp pack --profile archive
vp pack --profile review
```

Perfis:

```yaml
pack_profiles:
  training:
    format: webdataset
    shard_size: 1024
    compression: zstd
    image_quality: original

  archive:
    format: tar.zst
    compression_level: 15
    include_raw: true
    include_metadata: true

  review:
    format: folder
    image_quality: 85
    max_resolution: 1600
    include_previews: true
```

### 4.9 Exportação

```bash
vp export --format coco --output exports/coco-v3
vp export --format yolo --output exports/yolo-v3
vp export --format webdataset --output exports/train-shards
```

### 4.10 Anotação

```bash
vp annotate prepare --target cvat --split unlabeled
vp annotate prepare --target label-studio --limit 1000
vp annotate ingest ./annotations-from-cvat --format cvat
vp annotate review
```

A ferramenta deve facilitar ciclos de anotação:

1. selecionar imagens não anotadas ou de baixa confiança
2. empacotar para CVAT/Label Studio
3. ingerir anotações de volta
4. validar
5. comparar com versão anterior
6. criar snapshot

---

## 5. Estrutura de Diretórios

Estrutura recomendada:

```text
dataset/
  visionpack.yaml

  .vp/
    db/
      index.duckdb
    objects/
      sha256/
        ab/
          cd/
            abcdef...
    snapshots/
      v1.json
      v2.json
    cache/
    logs/

  assets/
    README.md

  annotations/
    README.md

  exports/
    coco/
    yolo/
    webdataset/

  reports/
    validation.html
    stats.html
```

Observação importante:

A pasta `.vp/objects` deve funcionar como content-addressable store. Arquivos são armazenados por hash, evitando duplicação.

---

## 6. Modelo de Dados

### 6.1 Asset

Um asset é uma imagem ou vídeo.

```json
{
  "id": "asset_01J...",
  "sha256": "abc123",
  "media_type": "image",
  "path": ".vp/objects/sha256/ab/cd/abc123",
  "original_path": "raw/img001.jpg",
  "width": 1920,
  "height": 1080,
  "channels": 3,
  "format": "jpeg",
  "size_bytes": 381022,
  "created_at": "2026-06-01T12:00:00Z",
  "metadata": {
    "camera": "line-4",
    "factory": "plant-a"
  }
}
```

### 6.2 Annotation

```json
{
  "id": "ann_01J...",
  "asset_id": "asset_01J...",
  "task": "detection",
  "format": "internal",
  "objects": [
    {
      "class_id": "scratch",
      "bbox": {
        "x": 120,
        "y": 80,
        "width": 240,
        "height": 140,
        "coordinate_system": "xywh_absolute"
      },
      "confidence": null,
      "attributes": {
        "occluded": false
      }
    }
  ],
  "source": {
    "type": "human",
    "tool": "cvat",
    "annotator": "operator_1"
  },
  "created_at": "2026-06-01T12:05:00Z"
}
```

### 6.3 Split

```json
{
  "id": "split_v3",
  "strategy": "stratified",
  "sets": {
    "train": ["asset_1", "asset_2"],
    "val": ["asset_3"],
    "test": ["asset_4"]
  },
  "locked": true,
  "created_at": "2026-06-01T12:10:00Z"
}
```

### 6.4 Snapshot

```json
{
  "version": "v3",
  "message": "added reviewed annotations",
  "created_at": "2026-06-01T12:15:00Z",
  "manifest_hash": "sha256...",
  "assets_hash": "sha256...",
  "annotations_hash": "sha256...",
  "splits_hash": "sha256...",
  "parent": "v2",
  "stats": {
    "assets": 12000,
    "annotations": 47000,
    "classes": 8
  }
}
```

---

## 7. Arquivo `visionpack.yaml`

Exemplo:

```yaml
name: factory-defects
version: 1

task: detection

classes:
  - id: scratch
    name: Scratch
  - id: dent
    name: Dent
  - id: stain
    name: Stain

storage:
  mode: content-addressed
  hash: sha256

validation:
  require_annotations: false
  allow_empty_images: true
  bbox:
    min_area_px: 4
    allow_out_of_bounds: false
  splits:
    prevent_leakage: true
  duplicates:
    exact: warn
    perceptual: off

splits:
  default:
    strategy: stratified
    train: 0.8
    val: 0.1
    test: 0.1
    stratify_by: class

exports:
  yolo:
    image_format: jpg
    normalized_coordinates: true
  coco:
    include_empty_images: true

pack_profiles:
  training:
    format: webdataset
    shard_size: 1024
    compression: zstd

  archive:
    format: tar.zst
    compression_level: 15
    include_metadata: true

annotation:
  preferred_tool: cvat
  review_required: true
```

---

## 8. Arquitetura Técnica

### 8.1 Linguagem

Recomendação inicial: **Python**.

Motivos:

- ecossistema forte em computer vision
- fácil integração com PyTorch, OpenCV, PIL, COCO tools
- bom para CLI e SDK
- adoção mais fácil por cientistas de dados

Bibliotecas sugeridas:

- `typer` para CLI
- `pydantic` para schemas
- `rich` para output no terminal
- `duckdb` para índice local
- `polars` para análises tabulares
- `pillow` para imagens
- `opencv-python` opcional
- `pyyaml` para config
- `zstandard` para compressão
- `orjson` para JSON rápido

Acelerações futuras podem ser feitas em Rust via `pyo3`, especialmente para hashing, diff e packing.

### 8.2 Módulos

Estrutura de código sugerida:

```text
visionpack/
  __init__.py

  cli/
    main.py
    commands/
      init.py
      import_.py
      validate.py
      stats.py
      split.py
      snapshot.py
      diff.py
      export.py
      pack.py
      annotate.py

  core/
    project.py
    manifest.py
    asset.py
    annotation.py
    snapshot.py
    split.py
    errors.py

  storage/
    object_store.py
    local_store.py
    hash.py
    materialize.py

  index/
    duckdb_index.py
    migrations.py
    queries.py

  formats/
    base.py
    yolo.py
    coco.py
    cvat.py
    label_studio.py

  validation/
    engine.py
    checks/
      images.py
      annotations.py
      bbox.py
      splits.py
      classes.py
      duplicates.py

  packing/
    profiles.py
    tar_zst.py
    webdataset.py

  diff/
    dataset_diff.py
    annotation_diff.py
    split_diff.py

  annotate/
    prepare.py
    ingest.py
    review.py

  reports/
    html.py
    json.py
    terminal.py

  training/
    torch_dataset.py
    datamodule.py

  plugins/
    registry.py
```

---

## 9. Interfaces

### 9.1 CLI

Interface principal.

Deve ser estável, limpa e scriptável.

### 9.2 Python SDK

Exemplo:

```python
from visionpack import Dataset

ds = Dataset.open(".")
ds.validate(strict=True)

snapshot = ds.snapshot("reviewed annotations")
train = ds.export(format="webdataset", split="train")
```

### 9.3 Integração com treino

Deve oferecer helpers para PyTorch:

```python
from visionpack.training import VisionPackDetectionDataset

dataset = VisionPackDetectionDataset(
    root=".",
    version="v3",
    split="train",
    transforms=my_transforms,
)
```

Mas o core não deve depender pesadamente de PyTorch.

### 9.4 GitHub Action

Futuro próximo:

```yaml
- uses: visionpack/validate-action@v1
  with:
    strict: true
    report: true
```

---

## 10. Estratégia de Versionamento

Versionamento deve funcionar por snapshots.

Cada snapshot referencia hashes de:

- manifesto
- lista de assets
- annotations
- splits
- transforms
- validações

Não tentar recriar Git do zero.

O armazenamento local pode funcionar assim:

- assets guardados por hash
- annotations normalizadas no índice
- snapshots como JSON
- exports gerados sob demanda
- arquivos derivados podem ser cacheados

Para datasets muito grandes, permitir modo `reference`, onde assets não são copiados, apenas indexados por caminho + hash.

---

## 11. Estratégia de Compressão

A compressão precisa ser orientada ao uso.

### 11.1 Archive

Para backup e transferência fria:

```bash
vp pack --profile archive
```

Formato:

- `.tar.zst`
- inclui manifesto
- inclui snapshots
- inclui índice exportável
- inclui assets e annotations

### 11.2 Training

Para treino eficiente:

```bash
vp pack --profile training
```

Formato recomendado:

- WebDataset shards
- `.tar` ou `.tar.zst`
- shards balanceados
- metadados por amostra
- split preservado

### 11.3 Review

Para revisão humana:

```bash
vp pack --profile review
```

Gera:

- imagens reduzidas
- previews
- labels em formato compatível com ferramenta de anotação
- pacote menor para enviar a anotadores

---

## 12. Boas Práticas de Dataset Embutidas

A ferramenta deve guiar o usuário para boas práticas sem ser paternalista.

Checks importantes:

- impedir train/test leakage
- alertar classes raras
- alertar mudanças bruscas de distribuição
- alertar duplicatas
- alertar labels inválidos
- alertar imagens com resolução muito fora da média
- preservar imagens vazias quando configurado
- permitir dataset com negative samples
- gerar relatório de cobertura de anotação
- registrar origem das labels
- diferenciar label humano, label sintético e pseudo-label

---

## 13. Fluxo de Trabalho Ideal

### 13.1 Criar dataset

```bash
vp init --name road-damage --task detection
vp import ./raw --format yolo
vp validate
vp stats
vp snapshot create -m "initial import"
```

### 13.2 Preparar anotação

```bash
vp annotate prepare --target cvat --where "annotation_count == 0" --limit 2000
```

### 13.3 Ingerir anotações

```bash
vp annotate ingest ./cvat-export.zip --format cvat
vp validate --strict
vp diff latest working
vp snapshot create -m "cvat batch 01 reviewed"
```

### 13.4 Exportar para treino

```bash
vp split create --strategy stratified --by class
vp pack --profile training
vp export --format yolo --output exports/yolo-v4
```

### 13.5 Registrar versão usada no modelo

```bash
vp snapshot tag v4 trained:model-2026-06-01
```

---

## 14. Funcionalidades Essenciais do MVP

Implementar nesta ordem:

1. `vp init`
2. `vp import` para YOLO
3. índice local com DuckDB
4. leitura de imagens e hashing
5. schema interno de annotation
6. `vp validate`
7. `vp stats`
8. `vp snapshot create/list/show`
9. `vp diff`
10. `vp export --format yolo`
11. `vp export --format coco`
12. `vp pack --profile archive`
13. `vp pack --profile training`

Não implementar UI web antes disso.

---

## 15. Design de Erros

Erros devem ser humanos e acionáveis.

Ruim:

```text
ValidationError: bbox invalid
```

Bom:

```text
Invalid bounding box in image img_0231.jpg

Class: scratch
Problem: x + width exceeds image width
Image size: 1280x720
Box: x=1200, y=200, width=300, height=80

Suggested fix:
- clamp boxes with: vp validate --fix bbox.clamp
- or inspect manually with: vp inspect img_0231.jpg
```

---

## 16. Diferenciais Competitivos

VisionPack deve se diferenciar por:

- foco específico em computer vision
- snapshots compreensíveis
- diff de datasets
- validação forte
- compressão orientada a treino/anotação/archive
- integração com formatos existentes
- CLI simples
- uso em CI
- preparação de pacotes para anotação
- rastreabilidade de dataset até modelo

Não vender como “data lake”, “annotation platform” ou “MLOps completo”.

Posicionamento:

> VisionPack is DatasetOps for Computer Vision.

---

## 17. Futuro Pós-MVP

Funcionalidades futuras:

- suporte a segmentation masks
- suporte a vídeos e tracking
- perceptual hashing para duplicatas visuais
- embeddings para near-duplicate detection
- UI local para revisão visual
- integração com S3/GCS/Azure
- integração com DVC/lakeFS
- lineage entre dataset e training runs
- dataset cards automáticos
- active learning queue
- pseudo-label management
- reviewer workflow
- plugin para CVAT
- dashboards HTML
- integração com FiftyOne

---

## 18. Critérios de Sucesso

O projeto é bem-sucedido se um usuário consegue:

1. importar um dataset YOLO bagunçado
2. descobrir problemas reais com `vp validate`
3. gerar estatísticas úteis
4. criar uma versão reproduzível
5. preparar um pacote para anotação
6. ingerir labels revisados
7. comparar duas versões
8. exportar para treino
9. compactar o dataset para storage ou pipeline
10. rastrear qual versão gerou determinado treino

---

## 19. Pedido para o Agente Implementador

Construa esse projeto com foco em qualidade de arquitetura, legibilidade e extensibilidade.

Priorize:

- schemas fortes
- CLI consistente
- testes de unidade para parsers e validators
- fixtures pequenas de datasets YOLO e COCO
- documentação clara
- erros acionáveis
- separação entre core, formatos, storage e CLI

Evite:

- criar web app cedo demais
- acoplar o core a PyTorch
- depender de uma estrutura fixa de pastas como fonte da verdade
- transformar a ferramenta em annotation tool completa
- inventar um formato fechado sem exportadores úteis
- otimizar prematuramente antes do fluxo básico funcionar

Resultado esperado do primeiro ciclo:

- pacote Python instalável
- comando `vp`
- suporte mínimo a YOLO detection
- importação, validação, stats, snapshot, diff e export
- README com exemplos reais
- testes cobrindo fluxos principais
