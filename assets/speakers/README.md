# Bundled speaker profiles

## en-female-1-neutral.json

- **Name:** en-female-1-neutral
- **Language:** English
- **Voice:** neutral female
- **Source:** `outetts` Python package v0.4.4 —
  `outetts/version/v3/default_speakers/json/en-female-1-neutral.json`
  (the only bundled V3 speaker in that release)
- **Reference text:** "The cat watched from the windowsill, tail flicking with quiet curiosity
  as the first snowflakes of winter began to fall, dusting the world in fragile white."
- **License:** Apache 2.0
  (`outetts` package METADATA: `License :: OSI Approved :: Apache Software License`)
- **Freely redistributable:** yes

### Schema (OuteTTS V3 interface_version=3)

```json
{
  "text": "<reference sentence>",
  "words": [
    {
      "word": "The",
      "duration": 0.20,
      "c1": [720, ...],
      "c2": [658, ...],
      "features": {"energy": 10, "spectral_centroid": 15, "pitch": 45}
    }
  ],
  "global_features": {"energy": 13, "spectral_centroid": 20, "pitch": 28},
  "interface_version": 3
}
```

`c1`/`c2` are DAC codebook-0/codebook-1 indices (0..1023), length = `round(duration * 75)` frames
at 75 fps (sr=24000 / hop=320).
