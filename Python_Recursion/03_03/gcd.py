"""
Python Recursion Video Course
Robin Andrews - https://compucademy.net/
"""
from functools import wraps


def trace(func):
    # Store function name, for later use
    func_name = func.__name__
    separator = '|  '  # Used in trace display

    # Set the current recursion depth
    trace.recursion_depth = 0

    @wraps(func)
    def traced_func(*args, **kwargs):
        # Display function call details
        print(f'{separator * trace.recursion_depth}|-- {func_name}({", ".join(map(str, args))})')
        # Begin recursing
        trace.recursion_depth += 1
        result = func(*args, **kwargs)
        # Exit current level
        trace.recursion_depth -= 1
        # Display return value
        print(f'{separator * (trace.recursion_depth + 1)}|-- return {result}')

        return result

    return traced_func


 # Adds higher directory to python modules path.


def gcd_recursive(a, b):
    if b == 0:
        return a
    else:
        return gcd_recursive(b, a % b)


gcd_recursive = trace(gcd_recursive)
print(gcd_recursive(32, 12))  # From slides
print(gcd_recursive(50, 15))
print(gcd_recursive(42, 28))
print(gcd_recursive(28, 42))
print(gcd_recursive(345, 766))  # Co-prime
