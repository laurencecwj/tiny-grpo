import sys
import random

count = int(sys.argv[1]) if len(sys.argv) > 1 else 20

symbols = ["-", "*", "+"]
results = []
for _ in range(count):
    _items_count = random.randint(2, 5)
    _items = [str(random.randint(-100, 100)) for _ in range(_items_count)]
    _syms = [symbols[random.randint(0, len(symbols)-1)] for _ in range(_items_count - 1)]
    _lst = list(zip(_items[:-1], _syms))
    _lst.append((_items[-1], ""))
    _result = []
    [_result.extend(_x) for _x in _lst if _x]
    _s = " ".join(_result)
    _r = eval(_s)
    print(f"{_s}")
    results.append(str(_r))

with open('results.txt', 'w') as wfl:
    wfl.write("\n".join(results))

