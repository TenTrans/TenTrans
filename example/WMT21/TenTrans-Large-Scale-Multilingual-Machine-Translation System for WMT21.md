# TenTrans Large-Scale Multilingual Machine Translation System for WMT21

http://statmt.org/wmt21/large-scale-multilingual-translation-task.html

**Small Track #2**: 5 South East Asian languages, 30 directions: Javanese, Indonesian, Malay, Tagalog, Tamil, English

The model and translation code we submitted can be found [here]().

## Statistics of all training data

We partially refer to m2m-100 data preprocessing(except for frequency cleaning).

https://github.com/pytorch/fairseq/tree/374fdc5cd94d361bb9b1089fe2c1d30a2eb15fdd/examples/m2m_100

```
# remove sentences with more than 50% punctuation
# deduplicate training data
# remove all instances of evaluation data from the training data
# apply spm
# length ratio cleaning
```

Our processing script is available [here]().

|           | En$\leftrightarrow$Id | En$\leftrightarrow$Jv | En$\leftrightarrow$Ms | En$\leftrightarrow$Ta | En$\leftrightarrow$Tl | Id$\leftrightarrow$Jv | Id$\leftrightarrow$Ms | Id$\leftrightarrow$Ta |
| --------- | --------------------- | --------------------- | --------------------- | --------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| No filter | 36.15M                | 1.52M                 | 7.43M                 | 1.19M                 | 6.97M                 | 0.75M                 | 4.23M                 | 0.46M                 |
| Filtered  | 33.67M                | 1.41M                 | 7.03N                 | 1.06M                 | 6.15M                 | 0.67M                 | 3.97M                 | 0.41M                 |
|           | Id$\leftrightarrow$Tl | Jv$\leftrightarrow$Ms | Jv$\leftrightarrow$Ta | Jv$\leftrightarrow$Tl | Ms$\leftrightarrow$Ta | Ms$\leftrightarrow$Tl | Ta$\leftrightarrow$Tl |                       |
| No filter | 2.56M                 | 0.41M                 | 0.06M                 | 0.74M                 | 0.33M                 | 1.27M                 | 0.53M                 |                       |
| Filtered  | 2.18M                 | 0.36M                 | 0.05N                 | 0.61M                 | 0.30M                 | 1.09M                 | 0.44M                 |                       |

## System Overview

### Base Systems

Our systems are based on the Transformer architecture as implemented in TenTrans.  We experiment with deeper encode layer (24) and larger feed-forward network size (4096) to provide reasonable performance. 

Because of the recent popularity of using large-scale pre-training models to fine-tune specific languages and tasks, we use the pre-trained model FLORES-101 released by the organizer to fine-tune on the bitext data. Note that to fine-tune FLORES-101 we train our models using FAIRSEQ.

### Forward-Translation and Back-Translation

For translation directions with more than 5 million bitext data, such as En$\leftrightarrow$Id, En$\leftrightarrow$Ms, En$\leftrightarrow$Tl, we separately train an individual model for each direction and use it for the pseudo-corpus generation. For other translation directions with less than 5 million bitext data, we use the baseline system of all language pairs jointly training for translating pseudo sentences.

This is results of the individual systems.

| System     | En$\rightarrow$Id | Id$\rightarrow$En | En$\rightarrow$Ms | Ms$\rightarrow$En | En$\rightarrow$Tl | Tl$\rightarrow$En |
| ---------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| Individual | 46.08             | 40.72             | 42.95             | 38.76             | 31.84             | 37.94             |

### In-domain Data Selection

we utilize pre-trained language model multilingual BERT to train a domain classifier for extracting in-domain sentences from authentic bilingual sentences. To train the domain classifier, we consider all available dev data as positive data, and randomly sample bilingual data as negative samples.

At the same domain test set, the domain classifier recognition accuracy is achieved at 93.97%. We select sentences predicted to be positive with a probability greater than threshold 0.7 to form an in-domain corpus.

The selection scores are [here]().

### Knowledge Distillation

We fine-tune the FLORES-101 model on five language pairs with En$\rightarrow$Ta, Id$\rightarrow$Ta, Jv$\rightarrow$Ta, Ms$\rightarrow$Ta, Tl$\rightarrow$Ta to produce an Any-to-Ta specific translation model. 

These five language pairs are chosen because they do not perform very well and have more room for improvement. We used this model as the teacher model to translate the training data of the five language pairs. The new data was then combined with data of other language pairs to train the student model.

### Gradual Fine-tuning

we use gradual fine-tuning combined with in-domain data selection. After training the domain classifier, authentic bilingual sentences with positive predictions and probabilities greater than the thresholds of 0.7, 0.8, 0.9, and 0.99 are selected to form in-domain corpora with different similarity degrees.

We started with a gradual fine-tuning on the domain-specific data selected at the 0.7 thresholds, followed by the 0.8 thresholds, and so on.

|      | En$\rightarrow$Id | En$\rightarrow$Jv | En$\rightarrow$Ms | En$\rightarrow$Ta | En$\rightarrow$Tl | Id$\rightarrow$En | Id$\rightarrow$Jv | Id$\rightarrow$Ms | Id$\rightarrow$Ta | Id$\rightarrow$Tl |
| ---- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| 0.7  | 3.83M             | 7.62K             | 1.15M             | 0.21M             | 0.36M             | 12.32M            | 0.05M             | 2.11M             | 0.23M             | 0.43M             |
| 0.8  | 3.44M             | 6.93K             | 1.05M             | 0.19M             | 0.34M             | 12.12M            | 0.05M             | 2.08M             | 0.22M             | 0.42M             |
| 0.9  | 2.82M             | 5.99K             | 0.89M             | 0.16M             | 0.30M             | 11.82M            | 0.04M             | 2.02M             | 0.22M             | 0.41M             |
| 0.99 | 1.24M             | 3.14K             | 0.44M             | 0.09M             | 0.17M             | 10.73M            | 0.03M             | 1.84M             | 0.20M             | 0.37M             |
|      | Jv$\rightarrow$En | Jv$\rightarrow$Id | Jv$\rightarrow$Ms | Jv$\rightarrow$Ta | Jv$\rightarrow$Tl | Ms$\rightarrow$En | Ms$\rightarrow$Id | Ms$\rightarrow$Jv | Ms$\rightarrow$Ta | Ms$\rightarrow$Tl |
| 0.7  | 59.99K            | 41.90K            | 15.82K            | 9.98K             | 14.77K            | 3.65M             | 2.20M             | 24.77K            | 0.18M             | 0.33M             |
| 0.8  | 57.12K            | 40.74K            | 14.83K            | 9.78K             | 14.29K            | 3.59M             | 2.17M             | 23.32K            | 0.18M             | 0.33M             |
| 0.9  | 53.05K            | 39.07K            | 13.76K            | 9.47K             | 13.61K            | 3.49M             | 2.11M             | 21.24K            | 0.17M             | 0.32M             |
| 0.99 | 41.59K            | 34.07K            | 10.47K            | 8.37K             | 11.39K            | 3.14M             | 1.91M             | 15.36K            | 0.16M             | 0.29M             |
|      | Ta$\rightarrow$En | Ta$\rightarrow$Id | Ta$\rightarrow$Jv | Ta$\rightarrow$Ms | Ta$\rightarrow$Tl | Tl$\rightarrow$En | Tl$\rightarrow$Id | Tl$\rightarrow$Jv | Tl$\rightarrow$Ms | Tl$\rightarrow$Ta |
| 0.7  | 0.72M             | 0.28M             | 17.60K            | 0.21M             | 0.20M             | 1.12M             | 0.43M             | 17.95K            | 0.31M             | 0.15M             |
| 0.8  | 0.71M             | 0.27M             | 16.83K            | 0.20M             | 0.19M             | 1.11M             | 0.42M             | 17.32K            | 0.30M             | 0.15M             |
| 0.9  | 0.69M             | 0.27M             | 15.73K            | 0.20M             | 0.19M             | 1.09M             | 0.41M             | 16.38K            | 0.30M             | 0.15M             |
| 0.99 | 0.63M             | 0.24M             | 12.49K            | 0.18K             | 0.16M             | 1.03M             | 0.38M             | 0.01M             | 0.28M             | 0.13M             |

## Train

Our training script is available [here]().

## Results

The main results are as follows:

| System           | Average BLEU |
| ---------------- | ------------ |
| Transformer      | 22.25        |
| + F&B            | 25.05        |
| + deep (24)      | 25.43        |
| FLORES-101       | 15.38        |
| + Fine-tuning    | 24.23        |
| + F&B            | 26.50        |
| + Data Selection | 27.24        |
| + Gradual FT     | 28.03        |
| + KD             | 28.15        |
| + Recover 12     | 28.32        |
| Averaging        | **28.94**    |

