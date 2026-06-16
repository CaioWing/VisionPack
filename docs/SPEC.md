---
title: Core Spec
parent: Internals
nav_order: 1
---

# VisionPack — Sample Envelope Spec (v1)

Este documento define o contrato canônico da plataforma e as regras que mantêm
ele estável e generalizável entre runtimes, transportes e hardwares diferentes.
O schema vive em `sample_envelope.proto`.

## O contrato como narrow waist

O `SampleEnvelope` é a única coisa que o core entende. Tudo a montante (como o
frame foi obtido) e a jusante (para onde vai) é adaptador. A regra inviolável:
nenhuma especificidade de plataforma pode vazar para dentro do schema do core.
Se você sentir vontade de adicionar um campo `deepstream_*` ou `if runtime ==`,
isso é sinal de adaptador ou capability faltando — não de exceção no contrato.

Por que Protobuf: codegen cross-language (você gera o stub para Python, C++,
Go, Rust a partir do mesmo `.proto`), wire format compacto para links de loja
metidos, e compatibilidade evolutiva embutida. O JSON canônico continua
disponível para debug.

## Regras de versionamento

A compatibilidade evolutiva é o que permite atualizar o core e os adapters em
ritmos diferentes — essencial numa frota de milhares de dispositivos com OTA.

Dentro de `visionpack.v1`, só mudanças aditivas. Concretamente: nunca renumere
um campo; nunca reutilize um número já usado; campo removido vira `reserved`
(reservando número e nome); campo novo entra com número novo e semântica
opcional. Como proto3 preserva campos desconhecidos, um core antigo ingere
envelopes de um adapter mais novo ignorando o que não reconhece (forward-compat),
e um adapter antigo continua válido para um core novo (backward-compat).

Mudança que quebra de fato — remover semântica de um campo existente, mudar tipo,
alterar unidade — exige bump de pacote para `visionpack.v2`, rodando lado a lado
com v1 durante a transição. O campo `schema_version` ("1.3") serve para debug,
filtro de log e gating de capability em runtime, separado da versão de pacote.

## Contrato de capabilities e degradação graciosa

Plataformas expõem sinais diferentes. O `AdapterCapabilities` é anunciado uma
vez no handshake (não por amostra) e diz ao core o que aquele adapter consegue
fornecer. O sampler então escolhe a política mais rica sustentável e cai para a
próxima quando o sinal não existe:

| Sinal disponível            | Política de sampling escolhida              |
|-----------------------------|---------------------------------------------|
| `provides_embedding`        | drift / novidade (distância OOD no espaço)  |
| `provides_confidence`       | incerteza (baixa confiança, entropia alta)  |
| nenhum dos dois             | reservoir com rate-limit (baseline)         |
| `can_redact_on_device`      | habilita política de privacidade (blur)     |

Heurísticas de falha (obstrução, nenhuma detecção, disagreement) entram quando o
adapter as expõe via `Trigger`. O ponto central: nunca *exigir* o sinal mais
rico — exigir embedding deixaria de fora a maioria dos usuários no dia um.

## Lazy hydration e dedup

O envelope carrega **referências**, não bytes. `BlobRef.sha256` é o endereço de
conteúdo e a chave de dedup: blobs idênticos entre frames, dispositivos, lojas
e até clientes sobem uma vez só. Na primeira passada o adapter sobe metadata +
embedding + thumbnail; o frame em resolução cheia só é hidratado quando a
curadoria na nuvem decide que aquela amostra vale anotação. Resultado: banda
proporcional ao valor, não ao volume.

## Privacidade

`Redaction` registra o que foi tratado no dispositivo antes de qualquer upload —
crítico em varejo, onde rostos de clientes não podem sair do edge crus. O core
pode recusar envelopes sem redação quando a política do site exigir.

## Exemplo (JSON canônico do proto3, camelCase)

```json
{
  "schemaVersion": "1.0",
  "envelopeId": "018f3a2b-7c41-7e9a-bd02-1f9c5a6e0d11",
  "capturedAt": "2026-06-06T14:22:31Z",
  "source": {
    "deviceId": "jetson-loja042-cam03",
    "siteId": "loja-042",
    "adapter": "deepstream",
    "adapterVersion": "0.3.1",
    "runtime": "deepstream-7.1/jetpack-6.0"
  },
  "model": {
    "name": "cart-detector",
    "version": "11n-int8",
    "artifactSha256": "9b1c...e7",
    "task": "TASK_TYPE_DETECTION"
  },
  "frame": {
    "sha256": "4f2a...c9",
    "mediaType": "image/jpeg",
    "sizeBytes": 184320,
    "width": 1920,
    "height": 1080
  },
  "thumbnail": { "sha256": "a07d...11", "mediaType": "image/webp", "width": 320, "height": 180 },
  "predictions": [
    {
      "label": "product",
      "labelIndex": 3,
      "confidence": 0.41,
      "trackId": "t-9182",
      "bbox": { "x": 0.512, "y": 0.337, "w": 0.084, "h": 0.121 }
    }
  ],
  "embedding": { "dim": 256, "model": "yolo-cls-head", "normalized": true, "vector": ["..."] },
  "triggers": [
    { "reason": "TRIGGER_REASON_LOW_CONFIDENCE", "score": 0.41 },
    { "reason": "TRIGGER_REASON_DRIFT", "score": 0.83, "detail": "cosine vs centroid" }
  ],
  "redaction": { "applied": true, "method": "REDACTION_METHOD_FACE_BLUR" },
  "attributes": { "aisle": "12", "shift": "evening" }
}
```

Repare que o `frame` aparece só como `sha256` + dimensões: a amostra acima
trafega em poucos KB, e o JPEG de 180 KB só sobe se a nuvem pedir.
