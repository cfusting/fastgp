import math


# 4 * x^4 + 3 * x^3 + 2 * x^2 + x
def mod_quartic(x):
    return x * (1 + x * (2 + x * (3 + x * 4)))


# Koza-1: x^4 + x^3 + x^2 + x
def quartic(x):
    return x * (1 + x * (1 + x * (1 + x)))


# Koza-2: x^5 - 2x^3 + x
def quintic(x):
    return x * (1 - x * x * (2 - x * x))


# Koza-3: x^6 - 2x^4 + x^2
def sextic(x):
    return x * x * (1 - x * x * (2 - x * x))


# x^7 - 2x^6 + x^5 - x^4 + x^3 - 2x^2 + x
def septic(x):
    return x * (1 - x * (2 - x * (1 - x * (1 - x * (1 - x * (2 - x))))))


# sum_{1}^9{x^i}
def nonic(x):
    return x * (1 + x * (1 + x * (1 + x * (1 + x * (1 + x * (1 + x * (1 + x * (1 + x))))))))


# x^3 + x^2 + x
def nguyen1(x):
    return x * (1 + x * (1 + x))


# x^5 + x^4 + x^3 + x^2 + x
def nguyen3(x):
    return x * (1 + x * (1 + x * (1 + x * (1 + x))))


# x^6 + x^5 + x^4 + x^3 + x^2 + x
def nguyen4(x):
    return x * (1 + x * (1 + x * (1 + x * (1 + x * (1 + x)))))


def nguyen5(x):
    return math.sin(x * x) * math.cos(x) - 1


def nguyen6(x):
    return math.sin(x) + math.sin(x * (1 + x))


def nguyen7(x):
    return math.log(x + 1) + math.log(x * x + 1)


def nguyen9(x, y):
    return math.sin(x) + math.sin(y * y)


def nguyen10(x, y):
    return 2 * math.sin(x) * math.cos(y)


def nguyen12(x, y):
    return x ** 4 - x ** 3 + (y ** 2 / 2.0) - y


def keijzer1(x):
    return 0.3 * x * math.sin(2 * math.pi * x)


def keijzer4(x):
    return x ** 3 * math.exp(-x) * math.cos(x) * math.sin(x) * (math.sin(x) ** 2 * math.cos(x) - 1)


def keijzer11(x, y):
    return (x * y) + math.sin((x - 1) * (y - 1))


def keijzer12(x, y):
    return x ** 4 - x ** 3 + (y ** 2 / 2.0) - y


def keijzer13(x, y):
    return 6 * math.sin(x) * math.cos(y)


def keijzer14(x, y):
    return 8.0 / (2 + x ** 2 + y ** 2)


def keijzer15(x, y):
    return (x ** 3 / 5.0) + (y ** 3 / 2.0) - x - y


def r1(x):
    return ((x + 1) ** 3) / (x ** 2 - x + 1)


def r2(x):
    return (x ** 5 - (3 * (x ** 3)) + 1) / (x ** 2 + 1)


def r3(x):
    return (x ** 6 + x ** 5) / (x ** 4 + x ** 3 + x ** 2 + x + 1)


def pagie1(x, y):
    return (1 / (1 + x ** -4)) + (1 / (1 + y ** -4))
