# 転置したときにreshapeはどういう挙動になるかの確認

import jax
from jax import numpy as jnp

array = jnp.asarray([[1,2],[3,4],[5,6]])

print(array)

array = jnp.transpose(array)

print(array)

array = jnp.reshape(array, (3,2))

print(array)