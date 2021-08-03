## WMT21 Multilingual Low-Resource Translation for Indo-European Languages shared task

We pay attention to the subtask of  **Wikipedia cultural heritage articles**.

- Task languages: **Romance languages**

- The translation direction: **from Catalan to Occitan, Romanian and Italian**（one-to-many）

  |               | oc  | ro   | it   |
| ------------- | ------ | ------- | ------- |
| **ca** | - | - | - |

- Related 4 high-resourced languages——Spanish, French and Portuguese data (+ English!) are allowed for training but translations are not evaluated.

## 1. Statistics of all training data

### 1.1 Low-resourced languages parallel data

- **(Task directions) ca-oc/ro/it **

|                     | ca-oc  | ca-ro   | ca-it   |
| ------------------- | ------ | ------- | ------- |
| Number of sentences | 138743 | 2169801 | 6345272 |
| Filtered            | 138741 | 2104099 | 5804556 |

- (Pairs between target low resource languages) **oc-ro-it**


|                     | it-ro   | it-oc  | oc-ro |
| ------------------- | ------- | ------ | ----- |
| Number of sentences | 7153087 | 121957 | 81406 |
| Filtered            | 6878705 | 121957 | 81406 |

### 1.2 High-low parallel data

- We have 4 related rich languages es/en/fr/pt allowed for training.

| Filtered | it       | oc     | ro       | ca      |
| -------- | -------- | ------ | -------- | ------- |
| **en**   | 22268782 | 59302  | 14534851 | 6963317 |
| **es**   | 4292528  | 35741  | 4152935  | 6451496 |
| **fr**   | 4728840  | 124055 | 1512046  | 6965692 |
| **pt**   | 15611591 | 24342  | 5556932  | 4553670 |

### 1.3  Monolingual data of low-resourced languages

|          | it       | ro       | oc     | ca      |
| -------- | -------- | -------- | ------ | ------- |
| Filtered | 38590424 | 13389667 | 225196 | 8297485 |

### 1.4 Data filtering

We  partially refer to m2m-100 data preprocessing(except for frequency cleaning) in FAIRSEQ。
https://github.com/pytorch/fairseq/tree/374fdc5cd94d361bb9b1089fe2c1d30a2eb15fdd/examples/m2m_100

## 2. Best single model(add links!)

We experiment with increasing network capacity by increasing embed dimension, FFN size, number of heads, and number of layers but find that deep and wide model architecture give us training hurdles. So that almost all subsequent models are based on the **base** Transformer architecture.

### 2.1 Baseline

- ca--oc/ro/it：as to the one-to-many multilingual model we using a single encoder/decoder shared for all languages.


Result：

|      | oc   | ro    | it    | AVG BLEU |
| ---- | ---- | ----- | ----- | -------- |
| ca   | 43.4 | 24.16 | 32.92 | 33.78    |

- Besides, we also experiment：
  - Training 1-1 dedicated bilingual models  
  - Joint training on ca/oc/ro/it simultaneously（many-to-many multilingual model ）

### 2.2 Pivot

Considering rich-low resource bilingual data，we train a **high(pivot) --> src** multilingual model for the purpose of back translation in advance. And the target side used for back translation is rich(pivot) in **high(pivot) --> tgt** bilingual data，so we get **src(pseudo)--tgt** bilingual data。

In the experiment, we train a high--ca multilingual model for BT, and back-translate rich resources data in high--oc/ro/it to get ca--oc/ro/it pseudo bilingual data.

<img src="/Users/yanghan/Library/Application Support/typora-user-images/image-20210713162329438.png" alt="image-20210713162329438" style="zoom:50%;" />

Note：to balance ture and pseudo data distribution, we oversample ca--ro true parallel data.

### 2.3 Back translation

Back-translation is an effective and commonly used data augmentation technique to incorporate monolingual data into a translation system.

- ca--it：since the parallel data in this direction is relatively rich compared to the other two directions, we train a bilingual BT model.
- ca--oc/ro：Because these two directions are facing a parallel data-scarce problem, so that we employ many-to-many multilingual model which trained on 4 low resource languages previously in baseline system to back translate, rather than using the dedicated bilingual model to BT.

Note: we do not employ data selection on generated data, but straightly use pseudo data to train a multilingual model from scratch, followed by fine-tuning on authentic parallel data.

### 2.4 High-low multilingual model

Training multiple language pairs together may result in transfer learning. Bring related high-resourced language pairs in the same typological language family should help gain a boost in low-resourced language pairs.

- We utilize **high--low** resource parallel data as well as pairs between low-resourced languages, to jointly train an 8-4 multilingual model. We get an **8-4 multilingual model**.
  - 8：means 4 high-resourced languages plus 4 low-resourced languages (en es pt fr + ca oc ro it)
  - 4：means 4 low-resourced languages (ca oc ro it)

Result：

|      | oc    | ro    | it    | AVG BLEU |
| ---- | ----- | ----- | ----- | -------- |
| ca   | 51.49 | 29.11 | 38.26 | 39.62    |

### 2.5 Pretrain

In addition to the above methods that utilize high resource-language paired with low-resourced languages, we also try experimenting with pretraining models that pre-trained on massive text to transfer knowledge for the target task.

We employ open resource pre-train model m2m-100, our experiments are based on the m2m-100 1.2B_last model:

| encoder-embed-dim | encoder-ffn-embed-dim | encoder-attention-heads | encoder layers |
| ----------------- | --------------------- | ----------------------- | -------------- |
| 1024              | 8192                  | 16                      | 24             |

- Parameters setting in fine tuning refers to mBART.
- ca--oc：firstly we employ multilingual fine-tune on the pre-trained model till the updates reach 20w/110w, after that we continue with bilingual fine-tune using authentic parallel data.
  - We try merely use authentic data or use authentic data plus pseudo data in multilingual fine-tune and finally choose the latter one.
- ca--ro：since the results after multilingual/bilingual fine-tune do not obtain promotion over original pre-train model, so we directly utilize pre-train model without fine-tune.

We exploit ca--oc、ca--ro data obtained through sentence level knowledge distillation and continue to train on 8-4 multilingual model trained above. Finally, we get a new model named **rich-m2m-KD**.

Result：

|      | oc    | ro    | it    | AVG BLEU |
| ---- | ----- | ----- | ----- | -------- |
| ca   | 65.18 | 32.85 | 36.19 | 44.74    |

## 3. Indomain fine tune

Dev and test dataset belong to the cultural heritage domain, whereas domains in training data are various, so domain adaptation is required.

- Pre-train model：bert_base_multilingual_cased

- Positive samples for training a classifier：combine dev sets of 3 languages ca/ro/it.

- We select parallel data predicted to be positive with a probability > 0.7 as the in-domain corpus, then we fine-tune on 8-4 model with a smaller learning rate. Finally, we get another \ model named **rich-indomain-ft**.

Result：

|      | oc   | ro   | it    | AVG BLEU |
| ---- | ---- | ---- | ----- | -------- |
| ca   | 56.6 | 28.3 | 38.74 | 41.21    |

## 4. Ensemble

- Primary：we combine rich-m2m-KD, rich-indomain-ft to be our ensemble model.


