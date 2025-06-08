#!/usr/bin/env python3

Dataset = __import__('1-dataset').Dataset

translator = Dataset()

# Show encoded example from training data
for source_text, target_text in translator.data_train.take(1):
    encoded_sample = translator.encode(source_text, target_text)
    print(encoded_sample)

# Show encoded example from validation data
for source_text, target_text in translator.data_valid.take(1):
    encoded_sample = translator.encode(source_text, target_text)
    print(encoded_sample)
