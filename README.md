# Comprehensive-Transformer-TTS - PyTorch Implementation

**A Non-Autoregressive Transformer** based TTS, supporting a family of SOTA transformers with supervised and unsupervised duration modelings. This project grows with the research community, **aiming to achieve the ultimate TTS**. Any suggestions toward the best Non-AR TTS are welcome :)

### Transformers
- [x] [Fastformer: Additive Attention Can Be All You Need](https://arxiv.org/abs/2108.09084) (Wu et al., 2021)
- [x] [Long-Short Transformer: Efficient Transformers for Language and Vision](https://arxiv.org/abs/2107.02192) (Zhu et al., 2021)
- [x] [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100) (Gulati et al., 2020)
- [x] [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) (Kitaev et al., 2020)
- [x] [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)

### Prosody Modelings (WIP)
- [x] [DelightfulTTS: The Microsoft Speech Synthesis System for Blizzard Challenge 2021](https://arxiv.org/abs/2110.12612) (Liu et al., 2021)
- [x] [Rich Prosody Diversity Modelling with Phone-level Mixture Density Network](https://arxiv.org/abs/2102.00851) (Du et al., 2021)

### Supervised Duration Modelings
- [x] [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558) (Ren et al., 2020)

### Unsupervised Duration Modelings
- [x] [One TTS Alignment To Rule Them All](https://arxiv.org/abs/2108.10447) (Badlani et al., 2021): We are finally freed from external aligners such as MFA! Validation alignments for LJ014-0329 up to 70K are shown below as an example.

  <p align="center">
      <img src="./img/LJ014-0329.gif" width="60%">
  </p>

### Transformer Performance Comparison on LJSpeech (1 TITAN RTX 24G / 16 batch size)
| Model | Memory Usage | Training Time (1K steps) |
| --- | ----------- | ----- |
|Fastformer (lucidrains')|10531MiB / 24220MiB|4m 25s
|Fastformer (wuch15's)|10515MiB / 24220MiB|4m 45s
|Long-Short Transformer|10633MiB / 24220MiB|5m 26s
|Conformer|18903MiB / 24220MiB|7m 4s
|Reformer|10293MiB / 24220MiB|10m 16s
|Transformer|7909MiB / 24220MiB|4m 51s
|Transformer_fs2|11571MiB / 24220MiB|4m 53s

Toggle the type of building blocks by
```yaml
# In the model.yaml
block_type: "transformer_fs2" # ["transformer_fs2", "transformer", "fastformer", "lstransformer", "conformer", "reformer"]
```

Toggle the type of prosody modelings by
```yaml
# In the model.yaml
prosody_modeling:
  model_type: "none" # ["none", "du2021", "liu2021"]
```

Toggle the type of duration modelings by
```yaml
# In the model.yaml
duration_modeling:
  learn_alignment: True # True for unsupervised modeling, and False for supervised modeling
```

# Quickstart

***DATASET*** refers to the names of datasets such as `LJSpeech` and `VCTK` in the following documents.

## Dependencies
You can install the Python dependencies with
```
pip3 install -r requirements.txt
```
Also, `Dockerfile` is provided for `Docker` users.

## Inference

You have to download the [pretrained models](https://drive.google.com/drive/folders/1xEOVbv3PLfGX8EgEkzg1014c9h8QMxQ-?usp=sharing) and put them in `output/ckpt/DATASET/`. The models are trained under unsupervised duration modeling with "transformer_fs2" building block.

For a **single-speaker TTS**, run
```
python3 synthesize.py --text "YOUR_DESIRED_TEXT" --restore_step RESTORE_STEP --mode single --dataset DATASET
```

For a **multi-speaker TTS**, run
```
python3 synthesize.py --text "YOUR_DESIRED_TEXT" --speaker_id SPEAKER_ID --restore_step RESTORE_STEP --mode single --dataset DATASET
```

The dictionary of learned speakers can be found at `preprocessed_data/DATASET/speakers.json`, and the generated utterances will be put in `output/result/`.


## Batch Inference
Batch inference is also supported, try

```
python3 synthesize.py --source preprocessed_data/DATASET/val.txt --restore_step RESTORE_STEP --mode batch --dataset DATASET
```
to synthesize all utterances in `preprocessed_data/DATASET/val.txt`.

## Controllability
The pitch/volume/speaking rate of the synthesized utterances can be controlled by specifying the desired pitch/energy/duration ratios.
For example, one can increase the speaking rate by 20 % and decrease the volume by 20 % by

```
python3 synthesize.py --text "YOUR_DESIRED_TEXT" --restore_step RESTORE_STEP --mode single --dataset DATASET --duration_control 0.8 --energy_control 0.8
```
Add ***--speaker_id SPEAKER_ID*** for a multi-speaker TTS.

# Training

## Datasets

The supported datasets are

- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/): a **single-speaker** English dataset consists of 13100 short audio clips of a female speaker reading passages from 7 non-fiction books, approximately 24 hours in total.
- [VCTK](https://datashare.ed.ac.uk/handle/10283/3443): The CSTR VCTK Corpus includes speech data uttered by 110 English speakers (**multi-speaker TTS**) with various accents. Each speaker reads out about 400 sentences, which were selected from a newspaper, the rainbow passage and an elicitation paragraph used for the speech accent archive.

Any of both **single-speaker TTS** dataset (e.g., [Blizzard Challenge 2013](https://www.synsig.org/index.php/Blizzard_Challenge_2013)) and **multi-speaker TTS** dataset (e.g., [LibriTTS](https://openslr.org/60/)) can be added following LJSpeech and VCTK, respectively. Moreover, **your own language and dataset** can be adapted following [here](https://github.com/keonlee9420/Expressive-FastSpeech2).

## Preprocessing

- For a **multi-speaker TTS** with external speaker embedder, download [ResCNN Softmax+Triplet pretrained model](https://drive.google.com/file/d/1F9NvdrarWZNktdX9KlRYWWHDwRkip_aP) of [philipperemy's DeepSpeaker](https://github.com/philipperemy/deep-speaker) for the speaker embedding and locate it in `./deepspeaker/pretrained_models/`.
- Run 
  ```
  python3 prepare_align.py --dataset DATASET
  ```
  for some preparations.

  For the forced alignment, [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/) (MFA) is used to obtain the alignments between the utterances and the phoneme sequences.
  Pre-extracted alignments for the datasets are provided [here](https://drive.google.com/drive/folders/1fizpyOiQ1lG2UDaMlXnT3Ll4_j6Xwg7K?usp=sharing). 
  You have to unzip the files in `preprocessed_data/DATASET/TextGrid/`. Alternately, you can [run the aligner by yourself](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/index.html).

  After that, run the preprocessing script by
  ```
  python3 preprocess.py --dataset DATASET
  ```

## Training

Train your model with
```
python3 train.py --dataset DATASET
```
Useful options:
- To use a [Automatic Mixed Precision](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html), append `--use_amp` argument to the above command.
- The trainer assumes single-node multi-GPU training. To use specific GPUs, specify `CUDA_VISIBLE_DEVICES=<GPU_IDs>` at the beginning of the above command.

# TensorBoard

Use
```
tensorboard --logdir output/log
```

to serve TensorBoard on your localhost.
The loss curves, synthesized mel-spectrograms, and audios are shown.

## LJSpeech

![](./img/tensorboard_loss_ljs.png)
![](./img/tensorboard_spec_ljs.png)
![](./img/tensorboard_audio_ljs.png)

## VCTK

![](./img/tensorboard_loss_vctk.png)
![](./img/tensorboard_spec_vctk.png)
![](./img/tensorboard_audio_vctk.png)

## Ablation Study
![](./img/tensorboard_loss_ljs_comparison.png)

| ID | Model | Block Type | Pitch Conditioning |
| --- | --- | ----------- | ----- |
|1|LJSpeech_transformer_fs2_cwt| `transformer_fs2` | continuous wavelet transform
|2|LJSpeech_transformer_cwt| `transformer` | continuous wavelet transform
|3|LJSpeech_transformer_frame| `transformer` | frame-level f0
|4|LJSpeech_transformer_ph| `transformer` | phoneme-level f0

Observations from
1. changing building block (ID 1~2):
  "transformer_fs2" seems to be more optimized in terms of memory usage and model size so that the training time and mel losses are decreased. However, the output quality is not improved dramatically, and sometimes the "transformer" block generates speech with an even more stable pitch contour than "transformer_fs2".
2. changing pitch conditioning (ID 2~4): There is a trade-off between audio quality (pitch stability) and expressiveness.
    - audio quality: "ph" >= "frame" > "cwt"
    - expressiveness: "cwt" > "frame" > "ph"

# Notes

- Both phoneme-level and frame-level variance are supported in both supervised and unsupervised duration modeling.
- Note that there are no pre-extracted phoneme-level variance features in unsupervised duration modeling.
- Unsupervised duration modeling in phoneme-level will take longer time than frame-level since the additional computation of phoneme-level variance is activated at runtime.
- Two options for embedding for the **multi-speaker TTS** setting: training speaker embedder from scratch or using a pre-trained [philipperemy's DeepSpeaker](https://github.com/philipperemy/deep-speaker) model (as [STYLER](https://github.com/keonlee9420/STYLER) did). You can toggle it by setting the config (between `'none'` and `'DeepSpeaker'`).
- DeepSpeaker on VCTK dataset shows clear identification among speakers. The following figure shows the T-SNE plot of extracted speaker embedding.

<p align="center">
    <img src="./preprocessed_data/VCTK/spker_embed_tsne.png" width="40%">
</p>

- For vocoder, **HiFi-GAN** and **MelGAN** are supported.

### Updates Log
- Mar.05, 2022 (v0.2.1): Fix and update codebase & pre-trained models with demo samples
  1. Fix variance adaptor to make it work with all combinations of building block and variance type/level
  2. Update pre-trained models with demo samples of LJSpeech and VCTK under "transformer_fs2" building block and "cwt" pitch conditioning
  3. Share the result of ablation studies of comparing "transformer" vs. "transformer_fs2" paired among three types of pitch conditioning ("frame", "ph", and "cwt")
- Feb.18, 2022 (v0.2.0): Update data preprocessor and variance adaptor & losses following [keonlee9420's DiffSinger](https://github.com/keonlee9420/DiffSinger) / Add various prosody modeling methods
  1. Prepare two different types of data pipeline in preprocessor to maximize unsupervised/supervised duration modelings
  2. Adopt wavelet for pitch modeling & loss
  3. Add fine-trained duration loss
  4. Apply `var_start_steps` for better model convergence, especially under unsupervised duration modeling
  5. Remove dependency of energy modeling on pitch variance
  6. Add "transformer_fs2" building block, which is more close to the original FastSpeech2 paper
  7. Add two types of prosody modeling methods
  8. Loss camparison on validation set:
    - LJSpeech - blue: v0.1.1 / green: v0.2.0
    <p align="center">
        <img src="./img/loss_comparison_ljs.png" width="80%">
    </p>

    - VCTK - skyblue: v0.1.1 / orange: v0.2.0
    <p align="center">
        <img src="./img/loss_comparison_vctk.png" width="80%">
    </p>
- Sep.21, 2021 (v0.1.1): Initialize with [ming024's FastSpeech2](https://github.com/ming024/FastSpeech2)

# Citation

Please cite this repository by the "[Cite this repository](https://github.blog/2021-08-19-enhanced-support-citations-github/)" of **About** section (top right of the main page).

# References
- [ming024's FastSpeech2](https://github.com/ming024/FastSpeech2)
- [wuch15's Fastformer](https://github.com/wuch15/Fastformer)
- [lucidrains' fast-transformer-pytorch](https://github.com/lucidrains/fast-transformer-pytorch)
- [lucidrains' long-short-transformer](https://github.com/lucidrains/long-short-transformer)
- [sooftware's conformer](https://github.com/sooftware/conformer)
- [lucidrains' reformer-pytorch](https://github.com/lucidrains/reformer-pytorch)
- [sagelywizard's pytorch-mdn](https://github.com/sagelywizard/pytorch-mdn)
- [keonlee9420's Robust_Fine_Grained_Prosody_Control](https://github.com/keonlee9420/Robust_Fine_Grained_Prosody_Control)
- [keonlee9420's Cross-Speaker-Emotion-Transfer](https://github.com/keonlee9420/Cross-Speaker-Emotion-Transfer)
- [keonlee9420's DiffSinger](https://github.com/keonlee9420/DiffSinger)
- [NVIDIA's NeMo](https://github.com/NVIDIA/NeMo): Special thanks to [Onur Babacan](https://github.com/babua) and [Rafael Valle](https://github.com/rafaelvalle) for unsupervised duration modeling.
