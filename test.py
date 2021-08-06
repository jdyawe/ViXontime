import datetime

import jqdatasdk as jds

import pandas as pd
import numpy as np

from time import perf_counter
import timeit

# ID = '18810592263'
# pwd = '592263'
# jds.auth(ID, pwd)
#
# count = jds.get_query_count()
# print(count)

N = 10000000
df = pd.DataFrame(np.random.rand(N, 2), columns=['a', 'b'])
print(df.head())

tic1 = perf_counter()

df['diffa'] = df['a'].diff()
df['diffb'] = df['b'].diff()

toc1 = perf_counter()

print('- - '*10)
print(f'time elapsed {toc1-tic1}')

print(df.head())

tic2 = perf_counter()

df['diffa1'] = df['a']-df['a'].shift(1)
df['diffb1'] = df['b']-df['b'].shift(1)

toc2 = perf_counter()

print('- - '*10)
print(f'time elapsed {toc2-tic2}')
print(df.head())

tic3 = perf_counter()

df['pcta'] = df['a'].pct_change()
df['pctb'] = df['b'].pct_change()

toc3 = perf_counter()

print('- - '*10)
print(f'time elapsed {toc3-tic3}')

print(df.head())

tic4 = perf_counter()

df['pcta1'] = (df['a'].diff())/df['a'].shift(1)
df['pctb1'] = (df['b'].diff())/df['b'].shift(1)

toc4 = perf_counter()

print('- - '*10)
print(f'time elapsed {toc4-tic4}')
print(df.head())