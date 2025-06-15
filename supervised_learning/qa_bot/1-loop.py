#!/usr/bin/env python3

while True:
    question = input("Q: ").strip()
    if question.lower() in ['exit', 'quit', 'goodbye', 'bye']:
        print("A: Goodbye")
        break
    print("A:")
