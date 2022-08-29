from functools import wraps


def bold(func):
    """Bold decorator"""
    @wraps(func)
    def wrapper():
        """return html bold tags"""
        result = '<b>' + func() + '</b>'
        return result
    return wrapper


def italics(func):
    """'''Italics decorator'''"""
    @wraps(func)
    def wrapper():
        """'''return html italics tags'''"""
        result = '<i>' + func() + '</i>'
        return result
    return wrapper


@bold
@italics
def printfib():
    '''return Fibonacci'''
    return 'Fibonacci'


print(printfib())


def style(func):
    """'''Italics decorator'''"""
    @wraps(func)
    def wrapper():
        """'''return html italics tags'''"""
        result = '<i><b>' + func() + '</i></b>'
        return result
    return wrapper


@style
def printfib():
    """'''return Fibonacci'''"""
    return 'Fibonacci'


print(printfib())
