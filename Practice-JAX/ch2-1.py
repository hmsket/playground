import numpy as np
import matplotlib.pyplot as plt

import jax, optax
from jax import random, numpy as jnp
from flax import linen as nn
from flax.training import train_state

key, key1, key2, key3 = random.split(random.PRNGKey(0), 4)
n0, mu0, variance0 = 20, [10, 11], 20
data0 = random.multivariate_normal(key1, jnp.asarray(mu0), jnp.eye(2)*variance0, jnp.asarray([n0]))
data0 = jnp.hstack([data0, jnp.zeros([n0, 1])])

n1, mu1, variance1 = 15, [18, 20], 22
data1 = random.multivariate_normal(key2, jnp.asarray(mu1), jnp.eye(2)*variance1, jnp.asarray([n1]))
data1 = jnp.hstack([data1, jnp.ones([n1, 1])])

data = random.permutation(key3, jnp.vstack([data0, data1]))
train_x, train_t = jnp.split(data, [2], axis=1)

class LogisticRegressionModel(nn.Module):
    @nn.compact
    def __call__(self, x, get_logits=False):
        x = nn.Dense(features=1)(x)
        if get_logits:
            return x
        x = nn.sigmoid(x)
        return x

key, key1 = random.split(key)
variables = LogisticRegressionModel().init(key1, train_x)

state = train_state.TrainState.create(
    apply_fn=LogisticRegressionModel().apply,
    params=variables['params'],
    tx=optax.adam(learning_rate=0.001)
)

@jax.jit
def loss_fn(params, state, inputs, labels):
    logits = state.apply_fn({'params': params}, inputs, get_logits=True)
    loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
    acc = jnp.mean(jnp.sign(logits) == jnp.sign(labels-0.5))
    return loss, acc

@jax.jit
def train_step(state, inputs, labels):
    f = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, acc), grads = f(state.params, state, inputs, labels)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, acc

loss_history, acc_history = [], []
for step in range(1, 10001):
    state, loss, acc = train_step(state, train_x, train_t)
    loss_history.append(jax.device_get(loss).tolist())
    acc_history.append(jax.device_get(acc).tolist())
    if step % 1000 == 0:
        print ('Step: {}, Loss: {:.4f}, Accuracy {:.4f}'.format(step, loss, acc), flush=True)