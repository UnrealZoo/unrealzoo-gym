# Uni-NaVid 

**A Video-based Vision-Language-Action Model for Unifying Embodied Navigation Tasks.** This project contains the finetuning and evaluation code of our RSS 2025 paper.


Contributors: [Jiazhao Zhang](https://jzhzhang.github.io/), Kunyu Wang, [Shaoan Wang](https://wsakobe.github.io/), Minghan Li, [Haoran Liu](https://yiconghong.me/), [Songlin Wei](https://songlin.github.io/), [Zhongyuan Wang](https://www.wangzhongyuan.com/), [Zhizheng Zhang](https://scholar.google.com/citations?user=X7M0I8kAAAAJ&hl=en), [He Wang](https://hughw19.github.io/)<br>

[[Paper & Appendices](https://arxiv.org/pdf/2412.06224)] [[Projece Page](https://pku-epic.github.io/Uni-NaVid/)]



<!-- https://github.com/user-attachments/assets/4ee1f806-03bb-4fcb-828e-2a7d9c6620c9



https://github.com/user-attachments/assets/304a512f-bfac-46e2-b293-f2e1e8b04f63 -->

![pipeline](./assets/uninavid.png)

## Release
- [x] Training Code
- [x] Offline Evaluation Code
- [x] Benchmark Evalation Code
    - [x] VLN-CE
    - [x] EVT-Bench
- [x] A small split of VLN-CE RxR data


## Cotents 

- [Install](#Install)
- [Preparation](#Preparation)
    - Model Preparation
    - Data Preparation
- [Train](#Train)
- [Evaluation](#Evaluation)
    - Offline Evaluation
    - Benchmark Evaluation
- [Citation](#Citation)
- [Acknowledgments](#Acknowledgments)


## Install

First, clone this repo:
```
git@github.com:jzhzhang/Uni-NaVid.git
```
Then install the Package and dependences:
```
conda create -n uninavid python=3.10 -y
conda activate uninavid
cd Uni-NaVid
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```
Finall, install the flash-attn package:
```
pip install flash-attn==2.5.9.post1
```

## Preparation

### Model

To train our model, you need to download the vision encoder and the language model. Below are the links to download the models in our paper:

| Model type | Model name | Download | 
|------|------|------|
| Encoder | EVA-CLIP | [ckpt](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth)|
| Pretrain model | Vicuna-7B | [ckpt](https://huggingface.co/lmsys/vicuna-7b-v1.5)|
| Finetuned model | Uni-NaVid (7B) | [ckpt](https://huggingface.co/Jzzhang/Uni-NaVid/tree/main/uninavid-7b-full-224-video-fps-1-grid-2)|

### Data

We provide a small subset of the data used in our paper to facilitate quick reproduction and customization with your own data. The data can be downloaded from [here](https://huggingface.co/Jzzhang/Uni-NaVid/tree/main/Nav-Finetune). The data is collcted from navigation tasks including the training splits of [VLN-CE](https://github.com/jacobkrantz/VLN-CE) R2R and RxR, [EVT-Bench](https://github.com/wsakobe/TrackVLA), [ObjectNav](https://arxiv.org/abs/2006.13171), [EQA](https://embodiedqa.org/). **Note that due to licensing restrictions, we did not use the [L3MVN](https://arxiv.org/pdf/2304.05501) method for ObjectNav limiation learning, which may result in a slight performance drop in ObjectNav evaluation.**

We recommend organizing your project directory as follows
```
Uni-NaVid
├── data
    ├── Nav-Finetune
        ├── nav_videos
        ├── open_uninavid_sampled_500.json
├── model_zoo
    ├── eva_vit_g.pth
    ├── <vicuna_weights> # optinoal, if you want to finetune from vicuna
    ├── <uninavid_weights> 
├── scripts
├── uninavid
├── test_cases # optinoal, if you want to offline evaluate uni-navid
```

## Train

Please set the `DATA_PATH` and `MODEL_PATH` in the `uninavid_stage_1.sh` and `uninavid_stage_2.sh` scripts to your data and model paths.

If you want to finetune from Vicuna-7B (make sure you collect sufficient data):
```
bash uninavid_stage_1.sh
```

If you want to  finetune based on Uni-NaVid:
```
bash uninavid_stage_2.sh
```


## Evaluation
During evaluation, the model leverages online token merging (`run_type=eval`), achieving an inference speed of approximately 5 Hz on a single A100 GPU. By employing more advanced techniques, such as quantization, the speed can be further enhanced.


### Offline Evaluation
We provide the offline evaluation code of Uni-NaVid on real-world videos, including a VLN sample `vln_1` and a tracking sample `tracking_1`. You can download the sample videos from [here](https://huggingface.co/Jzzhang/Uni-NaVid/tree/main/test_cases).

```
python offline_eval_uninavid.py test_cases/vln_1 Ourpur_dir # or test_cases/tracking_1
```
https://github.com/user-attachments/assets/31592c56-8369-4389-994f-f64b151ebb59

(move to the chair, then turn left and move forward to the humanoid robot and stop.)

https://github.com/user-attachments/assets/5ae851e0-d7fd-4b29-8501-05715febfc47

(follow the man with black top and brown pants.)



### Benchmark Evaluation 
We provide the evaluation code of Uni-NaVid on VLN-CE R2R/RxR and EVT Bench. 

Find the **VLN-CE benchmark** evaluation code [here](https://github.com/jzhzhang/NaVid-VLN-CE).

| Evaliation Benchmark |  TL  |  NE  |  OS  |  SR  |  SPL |
|----------------------|:----:|:----:|:----:|:----:|:----:|
| Uni-NaVid VLN-CE R2R Val.      | 9.22 | 4.96 | 57.4 | 51.8 | 47.7 |
| Uni-NaVid VLN-CE RxR Val.      | 18.4 | 5.67 | 66.4 | 56.1 | 44.5 |

Find the **EVT-bench** evaluation code [here](https://github.com/wsakobe/TrackVLA).

| Evaliation Benchmark |  SR  |  TR  |  CR  | 
|----------------------|:----:|:----:|:----:|
| Uni-NaVid EVT-Bench STT  | 53.3 | 67.2 | 12.6 | 
| Uni-NaVid EVT-Bench DT  | 31.9 | 50.1 | 21.3 | 
| Uni-NaVid EVT-Bench AT   | 15.8 | 41.5 | 26.5 | 


## Citation
If you find this work useful for your research, please consider citing:
```
@article{zhang2024uni,
    title={Uni-NaVid: A Video-based Vision-Language-Action Model for Unifying Embodied Navigation Tasks},
    author={Zhang, Jiazhao and Wang, Kunyu and Wang, Shaoan and Li, Minghan and Liu, Haoran and Wei, Songlin and Wang, Zhongyuan and Zhang, Zhizheng and Wang, He},
    journal={Robotics: Science and Systems},
    year={2025}
}
```



## Acknowledgments
Our code is based on [LLaMA-VID](https://github.com/dvlab-research/LLaMA-VID) and [NaVid](https://github.com/jzhzhang/NaVid-VLN-CE). 

This is an open-source version of Uni-NaVid, some functions have been rewritten to avoid certain license. 

If you have any questions, feel free to email Jiazhao Zhang at zhngjizh@gmail.com.
