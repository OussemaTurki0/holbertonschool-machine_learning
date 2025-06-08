#!/usr/bin/env python3

Dataset = __import__('0-dataset').Dataset

corpus = Dataset()

# Preview a sample from the training dataset
for french, english in corpus.data_train.take(1):
    print(french.numpy().decode('utf-8'))
    print(english.numpy().decode('utf-8'))

# Preview a sample from the validation dataset
for french, english in corpus.data_valid.take(1):
    print(french.numpy().decode('utf-8'))
    print(english.numpy().decode('utf-8'))

# Display tokenizer types
print(type(corpus.tokenizer_pt))  # Rename to tokenizer_fr if your Dataset uses French
print(type(corpus.tokenizer_en))
