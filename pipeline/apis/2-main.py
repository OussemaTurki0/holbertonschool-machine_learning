#!/usr/bin/env python3
"""
Test file for 2-user_location.py
"""

import os

if __name__ == '__main__':
    # Example GitHub user URL
    test_url = "https://api.github.com/users/holbertonschool"
    
    print("Testing valid user:")
    os.system(f"./2-user_location.py {test_url}")
    
    print("\nTesting invalid user:")
    os.system("./2-user_location.py https://api.github.com/users/thisuserdoesnotexistprobably123456")

    print("\nTesting without argument:")
    os.system("./2-user_location.py")
