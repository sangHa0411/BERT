# BERT


# Data Structure
```
.
├── dataset.py
├── encoder.py
├── loader.py
├── Log
├── masking.py
├── model.py
├── preprocessor.py
├── README.md
├── scheduler.py
├── Tokenizer
│   ├── make_tokenizer.py
│   └── tokenizer.py
└── train.py
```

# Dependencies
  1. pandas : '1.1.4'
  2. numpy : '1.19.2' 
  3. torch : '1.9.0+cu102'
  4. konlpy : '0.5.2'

# Tokenizer Specification
  1. Subword Tokenizer : BPE
  2. Sentencepiece
  3. Vocab size : 35000
  4. Special Token
      * PAD = 0
      * UNK = 1
      * SOS = 2
      * EOS = 3
      * CLS = 4
      * SEP = 5
      * MASK = 6
      * IGNORE_INDEX = -100

# Model Configuration
  1. BERT - Transformer Encoder Architecture
  2. Layer size : 12
  3. Embedding size : 768
  4. Hidden size : 3072
  5. Head size : 12
  6. Sequence size : 512
  7. DropOut Rate : 1e-1
  8. LayerNormalization : 1e-6

# Model Training
  1. SOP
        1. Forward Order Sentences
            > 1. Previous Sentence
            > 2. Next Sentence
        2. Backward Order Sentences
            > 1. Next Sentence
            > 2. Previous Sentence
        3. Random Sampled Sentences
            > 1. Sampled Sentence1
            > 2. Sampled Sentence2
  2. Masking
        1. Size : 15% of token list
            > 1. 80% : [MASK] token
            > 2. 10% : Random token
            > 3. 10% : Not changed
  3. Loss
        1. Predict SOP Label : 3 classes 
        2. Predict original token of [MASK] Token : 35000 classes
            * ignore which label is Token.IGNORE_INDEX

# Training Configuration
  1. Epoch : 30
  2. Warmup staps : 10000
  3. Optimizer : Adam
        1. Betas : (0.9, 0.999)
        2. Weight Decay : 1e-2
  4. Batch size : 64

# Source
  1. 모두의 말뭉치 : https://corpus.korean.go.kr
  2. 2020 신문 데이터
