def swapTwoVariables(x: int, y: int):
    x = x + y   # x + y
    y = x - y   # x
    x = x - y   # x + y - x
    return x, y
