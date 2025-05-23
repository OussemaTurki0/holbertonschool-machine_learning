#!/usr/bin/env python3
"""
Test file for the n-gram BLEU score implementation.
"""

ngram_bleu = __import__('1-ngram_bleu').ngram_bleu

# Example 1 - bigram (n=2)
references = [["the", "cat", "is", "on", "the", "mat"],
              ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["the", "cat", "is", "on", "the", "mat"]
print("BLEU score (bigram, Example 1):", ngram_bleu(references, sentence, 2))

# Example 2 - trigram (n=3)
references = [["this", "is", "a", "test", "example"]]
sentence = ["this", "is", "a", "test", "sample"]
print("BLEU score (trigram, Example 2):", ngram_bleu(references, sentence, 3))

# Example 3 - no match
references = [["hello", "world"]]
sentence = ["goodbye", "earth"]
print("BLEU score (bigram, No match):", ngram_bleu(references, sentence, 2))

# Example 4 - perfect match
references = [["the", "quick", "brown", "fox"]]
sentence = ["the", "quick", "brown", "fox"]
print("BLEU score (4-gram, Perfect match):", ngram_bleu(references, sentence, 4))

# Example 5 - longer reference, shorter sentence
references = [["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]]
sentence = ["the", "quick", "brown", "fox"]
print("BLEU score (bigram, Shorter sentence):", ngram_bleu(references, sentence, 2))
