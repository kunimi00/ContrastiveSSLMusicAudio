# Contrastive Self-supervised Learning of Music Audio

PyTorch implementation of [Towards Proper Contrastive Self-supervised Learning Strategies for Music Audio Representation](https://) by Jeong Choi et al.


In this work, we focuse on assessing the potential of self-supervised music embeddings as a general representation. We set up experiments to compare the performance in various MIR tasks between different self-supervision strategies. We investigate to what extent we can benefit from music audio representations learned from some of widely used contrastive learning schemes by analyzing the results on three different MIR tasks (instrument classification, genre classification, and music recommendation) which are considered to represent different aspects of music similarity.
Our experiments are set up using contrastive learning algorithms with variations in input / target instance settings and model architectures, which are designed to capture different levels the music semantic - global or regional information. Our strategies are categorized in the following table.

<img width="413" alt="strategies" src="https://user-images.githubusercontent.com/7988421/160309674-7e96edfa-04ad-4496-b966-733e5a55193b.png">

We then use the trained models as feature extractors and evaluate on different MIR tasks, where each task represents a certain abstraction level of music audio information. We compare the self-supervised embeddings with MFCCs which has long been a solid baseline feature in audio classification tasks.

## Transfer Learning (Linear Probing) Results
<img width="844" alt="results" src="https://user-images.githubusercontent.com/7988421/160309643-f9b63244-cf19-4480-a494-2b7faad76b23.png">


## Training
### 1. Pre-training
Run the following command to pre-train the model on the FMA_small dataset.
```
python main.py --USE_YAML_CONFIG 0
```

### 2. Inference audio embedding
To test a trained model, make sure to set the `LOAD_WEIGHT_FROM`, or specify it as an argument:
```
python main.py --MODE inference --LOAD_WEIGHT_FROM SomeCheckpointPath
```

## Linear evaluation
To be updated.


