import jax
from jax import numpy as jnp


@jax.jit
def plus1(idx, x):
    return x + 1

plus1_jit = jax.jit(plus1)

x = 0
N = 1000

ans = jax.lax.fori_loop(0, N, plus1_jit, x) # 0からN-1まで，plus1_jit()を実行する．引数の初期値はx．
print(ans)
