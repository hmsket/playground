import numpy as np
import matplotlib.pyplot as plt

import jax, optax
from jax import random, numpy as jnp
from flax import linen as nn
from flax.training import train_state


train_t = jnp.asarray([5.2, 5.7, 8.6, 14.9, 18.2, 20.4, 25.5, 26.4, 22.8, 17.5, 11.1, 6.6])
train_t = train_t.reshape([12, 1])

train_x = np.empty([12, 4])
for i in range(4):
    for month in range(12):
        train_x[month][i] = np.power(month+1, i+1)
train_x = jnp.asarray(train_x, dtype=jnp.int32)

class TemperatureModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        y = nn.Dense(features=1)(x)
        return y

key, key1 = random.split(random.PRNGKey(0))
variables = TemperatureModel().init(key1, train_x)

state = train_state.TrainState.create(
    apply_fn = TemperatureModel().apply,
    params = variables['params'],
    tx = optax.adam(learning_rate = 0.001)
)

@jax.jit
def loss_fn(params, state, inputs, labels):
    predicts = state.apply_fn({'params': params}, inputs)
    loss = optax.l2_loss(predicts, labels).mean()
    return loss

@jax.jit
def train_step(state, inputs, labels):
    f = jax.value_and_grad(loss_fn)
    loss, grads = f(state.params, state, inputs, labels)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

loss_history = []
for step in range(1, 101):
    state, loss_val = train_step(state, train_x, train_t)
    loss_history.append(jax.device_get(loss_val).tolist())
    if step % 1000 == 0:
        print(f'Step: {step}, Loss: {loss_val}')

plt.figure()
plt.plot(loss_history)
plt.show()