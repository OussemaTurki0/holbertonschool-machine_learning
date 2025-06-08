#!/usr/bin/env python3

Dataset = __import__('2-dataset').Dataset

dataset = Dataset()

# Display one example pair from training set
for input_text, output_text in dataset.data_train.take(1):
    print(input_text, output_text)

# Display one example pair from validation set
for input_text, output_text in dataset.data_valid.take(1):
    print(input_text, output_text)
