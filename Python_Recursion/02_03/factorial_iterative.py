"""
Python Recursion Video Course
Robin Andrews - https://compucademy.net/
"""


def factorial_iterative_while(n):  # Condition-controlled version
    results = 1
    while n >= 1:
        results *= n
        n -= 1

    return results
    # 5 * 4 * 3 * 2 * 1


# Let's do some basic testing
assert factorial_iterative_while(4) == 24
assert factorial_iterative_while(6) == 720
assert factorial_iterative_while(1) == 1
assert factorial_iterative_while(0) == 1
assert factorial_iterative_while(-7) == 1
assert factorial_iterative_while(50) == 30414093201713378043612608166064768844377641568960512000000000000
