#!/usr/bin/env python3

question_answer = __import__('0-qa').question_answer

def answer_loop(reference):
    """Answers questions interactively from a single reference"""
    while True:
        question = input("Q: ").strip()
        if question.lower() in ['exit', 'quit', 'goodbye', 'bye']:
            print("A: Goodbye")
            break

        answer = question_answer(question, reference)
        if answer:
            print("A:", answer)
        else:
            print("A: Sorry, I do not understand your question.")
