# Checkpoint contract

Every `pretrained_model` directory must contain at least `config.json`,
`model.safetensors`, `policy_preprocessor.json`, and
`policy_postprocessor.json`. Treat them as one atomic artifact.

Before deployment, compare the following values with the live pipeline:

- policy type and fused/DINO-only/DeFM-only mode;
- input feature names, dimensions, and wrist-camera order;
- state/action dimension, ordering, units, and normalization statistics;
- top and wrist token grids;
- FastFS input width/height, valid iterations, scale, maximum disparity, and
  maximum metric depth;
- action horizon, observation steps, and default action steps;
- local DINO and DeFM model identities.

`stereo-patch-infer --checkpoint ...` prints this contract. Local DINO/DeFM
paths serialized on a training machine can be overridden during loading; model
architecture and feature shape fields cannot be changed safely.

For a private Hugging Face model repository, upload the complete
`pretrained_model` directory. Authenticate through the Hugging Face CLI or
`HF_TOKEN`; never put a token in a script, manifest, Git remote, or commit.
