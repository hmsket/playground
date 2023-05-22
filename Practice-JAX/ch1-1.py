import numpy as np
import matplotlib.pyplot as plt

import jax
from jax import random, numpy as jnp


train_t = jnp.asarray([5.2, 5.7, 8.6, 14.9, 18.2, 20.4, 25.5, 26.4, 22.8, 17.5, 11.1, 6.6])
train_t = train_t.reshape([12, 1])

train_x = np.empty([12, 5])
for i in range(5):
    for month in range(12):
        train_x[month][i] = np.power(month+1, i)
train_x = jnp.asarray(train_x, dtype=jnp.int32)

key, key1 = random.split(random.PRNGKey(0))
w = random.normal(key1, [5, 1])

@jax.jit
def predict(w, x):
    y = jnp.matmul(x, w)
    return y

@jax.jit
def loss_fn(w, train_x, train_t):
    y = predict(w, train_x)
    errors = jnp.power((y - train_t), 2)
    loss = jnp.mean(errors)
    return loss

grad_loss = jax.jit(jax.grad(loss_fn))

lr = 1e-8 * 1.4
for step in range(50000):
    grads = grad_loss(w, train_x, train_t)
    w = w - lr * grads
    if (step+1) % 1000 == 0:
        loss_val = loss_fn(w, train_x, train_t)
        print(f'Step: {step+1}, Loss: {loss_val}')
