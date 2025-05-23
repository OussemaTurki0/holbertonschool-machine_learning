#!/usr/bin/env python3
"""
Test file for the unigram BLEU score implementation.
"""

uni_bleu = __import__('0-uni_bleu').uni_bleu

# Example 1
references = [["the", "cat", "is", "on", "the", "mat"],
              ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["the", "cat", "the", "cat", "on", "the", "mat"]
print("BLEU score (Example 1):", uni_bleu(references, sentence))

# Example 2
references = [["this", "is", "a", "test"]]
sentence = ["this", "is", "a", "test"]
print("BLEU score (Perfect match):", uni_bleu(references, sentence))

# Example 3
references = [["this", "is", "a", "test"]]
sentence = ["completely", "different", "words"]
print("BLEU score (No match):", uni_bleu(references, sentence))

# Example 4
references = [["the", "quick", "brown", "fox"],
              ["the", "fast", "brown", "fox"]]
sentence = ["the", "quick", "brown", "fox"]
print("BLEU score (Partial match):", uni_bleu(references, sentence))
