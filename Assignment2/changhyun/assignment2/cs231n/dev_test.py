def sum(x, a1, b1, c1):
    return x + a1 + b1 + c1

if __name__ == '__main__':
    x = 0
    a1 = 1
    b1 = 2
    c1 = 3
    f = lambda a: sum(x, a1, b1, c1)
    print(f(10000))