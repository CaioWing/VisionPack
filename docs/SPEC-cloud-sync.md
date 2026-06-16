---
title: Cloud Sync Spec
parent: Internals
nav_order: 2
---

# VisionPack — Cloud Sync Spec (v1)

Como o `vp sync` associa dados em object stores (S3, GCS) **sem baixar o dataset
inteiro** e **sem duplicar bytes**, mantendo a garantia de integridade e
reprodutibilidade que o resto da ferramenta promete.

Escopo da v1: **same-provider** (source e target no mesmo provedor — tudo S3 ou
tudo GCS). Cross-cloud (S3↔GCS) fica para um adapter de transferência futuro.

## O princípio

A identidade de um asset é **sempre o `sha256` do conteúdo** — igual ao caminho
local. Não existe identidade alternativa por etag/crc32c. Toda a complexidade que
um modelo "identidade por fingerprint" traria (estados provisórios, staging,
verificação, reconciliação) é evitada por uma única decisão:

> O `sha256` é calculado lendo cada objeto **exatamente uma vez**, e **nunca mais**.

Não há como endereçar conteúdo sem ver o conteúdo. A meta nunca foi "nunca ler";
é **não reler** no re-sync e **não persistir** localmente quando não for pedido.

## A regra do etag

O insight que mantém isso simples e robusto:

> `etag`/`crc32c` é confiável como **"esse objeto mudou?"** (mesma chave) e frágil
> como **"que objeto é esse?"** (entre chaves). Usamos só o primeiro.

Mesma chave + mesmo etag ⇒ conteúdo inalterado (garantido em S3 e GCS). Isso é
tudo que o re-sync precisa. **O etag nunca é comparado entre chaves diferentes** —
então ambiguidade de multipart, objetos cifrados (SSE-KMS) e colisão de crc32c
**nunca decidem nada**. Eles não são identidade; são um sino de "mudou".

## Fluxo do `vp sync`

1. Lista o metadata do(s) source(s) e do target (só nomes, `size`, `etag`) — zero
   corpo.
2. Para cada objeto, consulta o cache `blob_cache(uri, etag, size) → sha256`:
   - **bate** ⇒ pula. Zero leitura, zero download.
   - **não bate** (novo ou mudou) ⇒ lê **uma vez em streaming**, computa o `sha256`
     na passada (sem tocar o disco), grava no cache.
3. Materializa os bytes conforme o `copy` da source (abaixo) e grava o `Asset` no
   índice (`sha256` agora conhecido; `width`/`height`/`phash` podem ficar NULL até
   um pass que precise de pixel).

Idempotente: re-rodar re-lista, os etags batem, o delta é vazio, nada é lido nem
copiado. Vários sources caindo no mesmo target deduplicam pela chave de conteúdo.

## Modos de cópia (`copy:`)

Same-provider, sem operação irreversível na v1:

| modo | bytes | acoplamento | uso |
|---|---|---|---|
| `copy` (default cloud) | `CopyObject`/`rewrite` **server-side** para `target/objects/sha256/<ab>/<cd>/<sha>` | target autossuficiente, dedup global | caso comum |
| `reference` | nenhum movimento; o índice aponta para o objeto do source | depende do source vivo | sources que você controla e quer custo zero |
| `ingest` | baixa para o CAS local `.vp/objects/` | offline/edge | trabalho local |

**`move` não existe na v1.** Era a única operação irreversível e a fonte da maior
parte do risco (apagar a origem com base num sinal não verificado). Quem precisa
drenar um bucket de staging usa `copy` + uma *lifecycle policy* da própria nuvem —
mais simples e sem risco do nosso lado.

Em `copy`, os bytes **não passam pelo cliente** (a cópia é server-side, dentro do
provedor). O cliente só leu o objeto uma vez, no passo 2, para o hash.

## Export

- **Local (`hardlink`, default quando o CAS é local):** o diretório de export
  aponta para os inodes do CAS — **zero bytes extras**. (Substitui o `shutil.copy2`
  incondicional de hoje, que duplica cada imagem exportada.)
- **Cloud (`manifest`):** escreve labels + `manifest.jsonl` de `(uri, label, set)`
  e, opcionalmente, monta `target/export/<set>/` por cópia server-side. O trainer
  streama do bucket; **zero bytes locais**.

Split é **network-free** nos dois casos — opera só sobre o índice (ids derivados de
`sha256`, classe primária, phash). Nenhum download para splitar.

## Por que é robusto

- **Identidade é sempre o hash real** ⇒ dedup exato, split reproduzível entre
  máquinas e caminhos de ingestão, proveniência honesta. Sem estado provisório,
  sem promoção tardia, sem reconciliação retroativa.
- **O sinal barato (etag) só é usado onde é 100% seguro** (mudança na mesma chave).
- **Nenhuma operação irreversível na v1** (`move` fica de fora).
- **Re-sync não relê** ⇒ o custo recorrente é uma listagem de metadata.

Custo honesto: a **primeira** vez que um objeto é visto, ele é lido uma vez
(streaming, sem persistir). Se isso pesar em escala, a otimização é um worker que
roda o hash **in-region na nuvem** e devolve só o `sha256` — mas isso é tier pago e
fica fora da v1. O OSS lê uma vez do cliente e pronto.

## Impacto no código (mínimo)

- `Asset.sha256` passa a ser preenchível-lazy (NULL até a primeira leitura para
  sources `reference`/cloud); nenhum tipo novo de identidade.
- `Resolver` ganha `stat(uri) -> {size, etag}` (metadata-only, fallback) e
  `server_copy(src_uri, dst_uri)`. `read_bytes` continua sendo o caminho de hash.
  `list_files` já devolve `size`+`etag` por objeto na **mesma listagem**
  (`find(detail=True)`), então o re-sync não faz um HEAD por objeto.
- Credenciais/região declaradas no `visionpack.yaml` (`credentials:`, `region:`)
  são repassadas ao filesystem do provedor via `storage_options` do fsspec; o
  default continua sendo auth de ambiente (env/instance-role).
- Nova tabela `blob_cache(uri, etag, size, sha256)` no índice SQLite.
- `CopyMode` ganha `reference` para cloud; `move` não entra.
- `target:` no `visionpack.yaml` (uri + layout content-addressed).

Sem staging, sem ledger, sem gate de verificação. É o `git`-para-cloud: hash uma
vez, content-address, reusa para sempre.

## Sequência de PRs

1. ✅ `Resolver.stat` + `FsspecResolver` (list/stat metadata-only) + `blob_cache`
   de re-sync. Habilita `vp sync --dry-run` sobre S3/GCS sem baixar nada.
2. ✅ `server_copy` + modos `copy`/`reference` + `target:` no yaml → sync
   cloud-internal ponta a ponta. (YOLO; COCO/imagefolder remoto ficam para depois.)
3. ✅ Export `hardlink` local + `manifest` cloud (`AssetMaterializer`): assets
   locais são hardlinkados do CAS (zero bytes extras); assets remotos viram
   `manifest.jsonl` de `(image, uri, ...)` sem baixar nada. Vale para os três
   exporters de diretório (yolo/coco/imagefolder); `pack` segue local-only.
