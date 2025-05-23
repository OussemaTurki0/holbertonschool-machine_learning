#!/usr/bin/env python3
"""
Test file for the cumulative n-gram BLEU score implementation.
"""

cumulative_bleu = __import__('2-cumulative_bleu').cumulative_bleu

# Example 1 - cumulative BLEU with n=4
references = [["the", "cat", "is", "on", "the", "mat"],
              ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["the", "cat", "is", "on", "the", "mat"]
print("Cumulative BLEU (n=4, Example 1):", cumulative_bleu(references, sentence, 4))

# Example 2 - exact match
references = [["this", "is", "a", "test"]]
sentence = ["this", "is", "a", "test"]
print("Cumulative BLEU (n=4, Perfect match):", cumulative_bleu(references, sentence, 4))

# Example 3 - no match
references = [["this", "is", "a", "test"]]
sentence = ["totally", "different", "sentence"]
print("Cumulative BLEU (n=4, No match):", cumulative_bleu(references, sentence, 4))

# Example 4 - short sentence, long reference
references = [["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]]
sentence = ["the", "quick", "brown", "fox"]
print("Cumulative BLEU (n=4, Shorter sentence):", cumulative_bleu(references, sentence, 4))

# Example 5 - partial match
references = [["I", "love", "machine", "learning", "and", "natural", "language", "processing"]]
sentence = ["I", "love", "deep", "learning", "and", "AI"]
print("Cumulative BLEU (n=3, Partial match):", cumulative_bleu(references, sentence, 3))
