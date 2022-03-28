# Contrastive Self-supervised Learning of Music Audio

PyTorch implementation of [Towards Proper Contrastive Self-supervised Learning Strategies for Music Audio Representation](https://) by Jeong Choi et al.


In this work, we focuse on assessing the potential of self-supervised music embeddings as a general representation. We set up experiments to compare the performance in various MIR tasks between different self-supervision strategies. We investigate to what extent we can benefit from music audio representations learned from some of widely used contrastive learning schemes by analyzing the results on three different MIR tasks (instrument classification, genre classification, and music recommendation) which are considered to represent different aspects of music similarity.
Our experiments are set up using contrastive learning algorithms with variations in input / target instance settings and model architectures, which are designed to capture different levels the music semantic - global or regional information. Our strategies are categorized in the following table.



We then use the trained models as feature extractors and evaluate on different MIR tasks, where each task represents a certain abstraction level of music audio information. We compare the self-supervised embeddings with MFCCs which has long been a solid baseline feature in audio classification tasks.


## Pre-train on your own folder of audio files
Run the following command to pre-train the model on a folder containing .wav files (or .mp3 files when editing `src_ext_audio=".mp3"` in `/datasets/audio.py`). You may need to convert your audio files to the correct sample rate first, before giving it to the encoder (which accepts `18,000Hz` per default).

```
python preprocess.py --dataset audio --dataset_dir ./directory_containing_audio_files

python main.py --dataset audio --dataset_dir ./directory_containing_audio_files
```


## Results



## Training
### 1. Pre-training
Run the following command to pre-train the model on the MSD dataset.
```
python main.py --dataset msd
```

### 2. Linear evaluation
To test a trained model, make sure to set the `checkpoint_path`, or specify it as an argument:
```
python linear_evaluation.py --checkpoint_path ./checkpoint_10000.pt
```

## Configuration


