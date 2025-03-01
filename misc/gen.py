import sys
import random
import json

MAX_ITEMS=5
MAX_DIGITS=6
COUNT=10000

symbols = ["-", "*", "+"]
results = []
datas = []
for _id in range(COUNT):
    _items_count = random.randint(2, MAX_ITEMS)
    _digits_count = random.randint(2, MAX_DIGITS)
    _items = [str(random.randint(-1*pow(10, _digits_count), 1*pow(10, _digits_count))) for _ in range(_items_count)]
    _digits = [len(str(abs(int(_x)))) for _x in _items]
    _syms = [symbols[random.randint(0, len(symbols)-1)] for _ in range(_items_count - 1)]
    _lst = list(zip(_items[:-1], _syms))
    _lst.append((_items[-1], ""))
    _result = []
    [_result.extend(_x) for _x in _lst if _x]
    _s = " ".join(_result)
    _r = eval(_s)
    # print(f"{_s}")
    results.append(str(_r))
    _data = {
        'id': str(_id),
        'question': _s,
        'answer': str(_r),
        'num_terms': _items_count, 
        'num_digits': max(_digits),
    }
    datas.append(_data)

with open('results.json', 'w') as wfl:
    for _d in datas:
        wfl.write(json.dumps(_d) + "\n")

# with open('results.txt', 'w') as wfl:
#     wfl.write("\n".join(results))

