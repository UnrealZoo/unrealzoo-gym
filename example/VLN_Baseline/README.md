# UnrealZoo VLN Baseline

This directory adapts VLN and visual navigation baselines to the UnrealZoo navigation environment. The common entry point is:

```bash
python example/VLN_Baseline/vln_baseline.py --method <method>
```

Supported methods:

- `uni_navid`: Uni-NaVid language-guided navigation.
- `vint`: ViNT image-goal navigation.
- `nomad`: NoMaD image-goal navigation. It shares the VisualNav dependency stack with ViNT.
- `streamvln`: StreamVLN language-guided navigation.

The script converts model outputs to UnrealZoo mixed actions:

- `forward`
- `left`
- `right`
- `stop`

## Environment

The examples below assume the project root is `/home/l/code/unrealzoo-gym` and the conda environment is `unrealzoo`.

```bash
cd /home/l/code/unrealzoo-gym
conda activate unrealzoo
```

If the Unreal Engine package is under the project-local `UnrealEnv` directory, pass:

```bash
--unreal-env ./UnrealEnv
```

Use `conda run --no-capture-output` when launching large models so loading progress is printed immediately.

## Install Dependencies

The three model families were developed with different Python stacks. A single shared environment can work, but version conflicts are possible. The commands below are the currently tested shared-environment setup.

### Uni-NaVid

Uni-NaVid depends on the LLaVA/LLaMA video stack and SentencePiece.

```bash
conda run -n unrealzoo pip install \
  imageio opencv-python \
  "transformers==4.51.0" "tokenizers>=0.21.0" \
  sentencepiece einops einops-exts timm shortuuid peft \
  decord fairscale scikit-learn
```

If Uni-NaVid fails with:

```text
TypeError: Descriptors cannot be created directly.
```

downgrade protobuf:

```bash
conda run -n unrealzoo pip install "protobuf==3.20.3"
```

or launch Uni-NaVid with:

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

Checkpoint directory expected by the example:

```text
example/VLN_Baseline/Uni-NaVid/model_zoo/uninavid-7b-full-224-video-fps-1-grid-2
```

### ViNT / NoMaD

ViNT and NoMaD come from `visualnav-transformer` and use image-goal navigation. They require a goal image through `--goal-image`.

```bash
conda run -n unrealzoo pip install \
  warmup-scheduler efficientnet-pytorch diffusers vit-pytorch pyyaml
```

Default checkpoint locations:

```text
example/VLN_Baseline/visualnav-transformer/deployment/model_weights/vint.pth
example/VLN_Baseline/visualnav-transformer/deployment/model_weights/nomad.pth
```

If a checkpoint is stored elsewhere, pass:

```bash
--visualnav-model-path /path/to/model.pth
```

### StreamVLN

StreamVLN uses a newer Qwen/LLaVA-video stack, so it needs a recent `transformers` and `accelerate`.

```bash
conda run -n unrealzoo pip install \
  "transformers==4.51.0" "tokenizers>=0.21.0" \
  "accelerate>=1.13.0" numpy-quaternion omegaconf
```

Optional StreamVLN packages may print warnings if missing:

```text
Please install pyav to use video processing functions.
Please install petrel_client to Client.
OpenCLIP not installed
```

These warnings do not block the UnrealZoo online example.

Default StreamVLN checkpoint directory:

```text
example/VLN_Baseline/StreamVLN/checkpoints/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world
```

Default SigLIP vision tower directory:

```text
example/VLN_Baseline/StreamVLN/checkpoints/siglip-so400m-patch14-384
```

The adapter automatically redirects StreamVLN's `google/siglip-so400m-patch14-384` config to the local SigLIP directory when `model.safetensors` exists there.

## Run Examples

### Uni-NaVid

```bash
conda run --no-capture-output -n unrealzoo python example/VLN_Baseline/vln_baseline.py \
  --method uni_navid \
  --model-path example/VLN_Baseline/Uni-NaVid/model_zoo/uninavid-7b-full-224-video-fps-1-grid-2 \
  --instruction "navigate to the gray car on your left back side" \
  --target BP_Car_Gray \
  -r \
  --unreal-env ./UnrealEnv
```

### ViNT

ViNT is not language-conditioned. `--instruction` is accepted by the common entry point, but navigation is driven by `--goal-image`.

```bash
conda run --no-capture-output -n unrealzoo python example/VLN_Baseline/vln_baseline.py \
  --method vint \
  --goal-image doc/figs/navigation/target_2.png \
  --instruction "navigate to the target" \
  --target BP_Car_Gray \
  -r \
  --unreal-env ./UnrealEnv
```

### NoMaD

NoMaD also uses image-goal navigation. Make sure `nomad.pth` exists first.

```bash
conda run --no-capture-output -n unrealzoo python example/VLN_Baseline/vln_baseline.py \
  --method nomad \
  --goal-image doc/figs/navigation/target_2.png \
  --instruction "navigate to the target" \
  --target BP_Car_Gray \
  -r \
  --unreal-env ./UnrealEnv
```

### StreamVLN

```bash
conda run --no-capture-output -n unrealzoo python example/VLN_Baseline/vln_baseline.py \
  --method streamvln \
  --instruction "navigate to the gray car on your left back side" \
  --target BP_Car_Gray \
  -r \
  --unreal-env ./UnrealEnv
```

During startup, StreamVLN should print progress like:

```text
[StreamVLN] loading tokenizer/config...
[StreamVLN] using local vision tower: ...
[StreamVLN] loading model weights, this can take several minutes...
[StreamVLN] ready.
```

## Common Options

```bash
--env-id UnrealNavigation-SuburbNeighborhood_Day-MixedColor-v0
--resolution 240 240
--max-steps 500
--forward-speed 100
--turn-speed 30
--offscreen
--display :0
```

Disable rendering by removing `-r`.

## Dependency Conflict Notes

Uni-NaVid and StreamVLN are the most likely to conflict because they expect different generations of the LLaVA/Transformers stack. The current shared environment uses:

```text
transformers==4.51.0
accelerate>=1.13.0
protobuf==3.20.3
```

If future changes reintroduce conflicts, the recommended long-term solution is to run model inference in separate conda environments and expose each model through a small subprocess or local HTTP/RPC service. Keep UnrealZoo and `env.step()` in the main `unrealzoo` environment, and isolate Uni-NaVid, VisualNav, and StreamVLN dependencies from each other.

## References

If you use these adapted baselines, please cite the original methods:

```bibtex
@article{zhang2024uni,
  title={Uni-NaVid: A Video-based Vision-Language-Action Model for Unifying Embodied Navigation Tasks},
  author={Zhang, Jiazhao and Wang, Kunyu and Wang, Shaoan and Li, Minghan and Liu, Haoran and Wei, Songlin and Wang, Zhongyuan and Zhang, Zhizheng and Wang, He},
  journal={Robotics: Science and Systems},
  year={2025}
}

@inproceedings{shah2023vint,
  title={ViNT: A Foundation Model for Visual Navigation},
  author={Shah, Dhruv and Sridhar, Ajay and Dashora, Nitish and Stachowicz, Kyle and Black, Kevin and Hirose, Noriaki and Levine, Sergey},
  booktitle={Conference on Robot Learning},
  pages={711--733},
  year={2023},
  organization={PMLR}
}
@inproceedings{sridhar2024nomad,
  title={Nomad: Goal masked diffusion policies for navigation and exploration},
  author={Sridhar, Ajay and Shah, Dhruv and Glossop, Catherine and Levine, Sergey},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={63--70},
  year={2024},
  organization={IEEE}
}

@article{wei2025streamvln,
  title={StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling},
  author={Wei, Meng and Wan, Chenyang and Yu, Xiqian and Wang, Tai and Yang, Yuqiang and Mao, Xiaohan and Zhu, Chenming and Cai, Wenzhe and Wang, Hanqing and Chen, Yilun and others},
  journal={arXiv preprint arXiv:2507.05240},
  year={2025}
}
```
