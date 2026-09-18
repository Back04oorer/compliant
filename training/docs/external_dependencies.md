# External dependencies

This repository does not redistribute datasets, checkpoints, or third-party
foundation-model source trees. Supply local paths to:

- [LeRobot](https://github.com/huggingface/lerobot) 0.4.4.
- [DINOv3](https://github.com/facebookresearch/dinov3) model
  `facebook/dinov3-vitb16-pretrain-lvd1689m` in a Transformers-compatible local
  directory. The Hugging Face model is gated: accept Meta's license and log in
  before downloading it.
- [DeFM](https://github.com/leggedrobotics/defm). Its model factory downloads
  pretrained weights from `leggedrobotics/defm` by default; pass
  `--defm-checkpoint` for a fully local/offline run.
- [Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo)
  and a compatible serialized checkpoint. The reference pipeline uses the
  upstream `20-30-48/model_best_bp2_serialize.pth` checkpoint; obtain it from
  the weights link in Fast-FoundationStereo's README.

Review and comply with each upstream project's license and model terms. The
Apache-2.0 license in this repository covers this integration code only.

The reference integration was tested with DeFM commit
`63ec5e1c1a9b280dcde9910b845f57e9224ebab5` and Fast-FoundationStereo commit
`a290ba04c1b3ad1ec41a33974a157b2917b624d4`. Later upstream revisions may work,
but should be smoke-tested before a long run.

Use the tested source revisions:

```bash
mkdir -p external

git clone https://github.com/leggedrobotics/defm.git external/defm
git -C external/defm checkout 63ec5e1c1a9b280dcde9910b845f57e9224ebab5
pip install -e external/defm --no-deps

git clone https://github.com/NVlabs/Fast-FoundationStereo.git \
  external/Fast-FoundationStereo
git -C external/Fast-FoundationStereo checkout \
  a290ba04c1b3ad1ec41a33974a157b2917b624d4
```

`requirements.txt` provides the dependencies needed on the Stereo-ACT path;
`--no-deps` prevents the DeFM editable install from replacing the deliberately
selected shared environment. FastFS and DINO/DeFM model weights must still be
downloaded under their upstream license terms.
