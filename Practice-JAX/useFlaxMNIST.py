import numpy as np
from tensorflow.keras.datasets import mnist

import jax, optax
from jax import random, numpy as jnp
from flax import linen as nn
from flax.training import train_state


class MyNet(nn.Module):
    @nn.compact
    def __call__(self, x, get_logits=False):
        nh = 5
        ks = 10
        x = x.reshape([1, 28, 28, 1]) # 次元合わせのおまじない
        x = nn.Conv(features=nh, kernel_size=(ks, ks), strides=1, padding='VALID', use_bias=False)(x)
        x_size = (29-ks)**2
        x = x.reshape([x_size, -1]) # flatten代わり
        zeros = jnp.zeros([1, nh])
        x = jnp.vstack([zeros, x])
        x = nn.softmax(x, axis=0)
        x = jnp.sum(x[1:], axis=0)
        x = nn.Dense(features=10)(x)
        if get_logits:
            return x
        x = nn.softmax(x)
        return x


def get_mnist_dataset():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape([-1, 784]).astype('float32') / 255
    test_images = test_images.reshape([-1, 784]).astype('float32') / 255
    train_labels = np.eye(10)[train_labels]
    test_labels = np.eye(10)[test_labels]
    return (train_images, train_labels), (test_images, test_labels)


@jax.jit
def loss_fn(params, state, inputs, labels):
    logits = state.apply_fn({'params': params}, inputs, get_logits=True)
    loss = optax.softmax_cross_entropy(logits, labels).mean()
    # logits = nn.softmax(logits)
    acc = jnp.mean(jnp.argmax(logits) == jnp.argmax(labels))
    return loss, acc


@jax.jit
def train_step(state, inputs, labels):
    f = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, acc), grads = f(state.params, state, inputs, labels)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, acc


def train_epoch(state, inputs, labels, eval):
    losses, accs = [], []
    for input, label in zip(inputs, labels):
        new_state, loss, acc = train_step(state, input, label)
        if not eval:
            state = new_state
        losses.append(jax.device_get(loss).tolist())
        accs.append(jax.device_get(acc).tolist())
    return state, np.mean(losses), np.mean(accs)


def fit(state, train_inputs, train_labels, test_inputs, test_labels, epochs):
    losses_train, accs_train = [], []
    losses_test, accs_test = [], []

    for epoch in range(1, epochs+1):
        # training
        state, loss_train, acc_train = train_epoch(state, train_inputs, train_labels, eval=False)
        losses_train.append(loss_train)
        accs_train.append(acc_train)

        # evaluation
        _, loss_test, acc_test = train_epoch(state, test_inputs, test_labels, eval=True)
        losses_test.append(loss_test)
        accs_test.append(acc_test)

        print ('Epoch: {}, Loss: {:.4f}, Accuracy: {:.4f} / '.format(epoch, loss_train, acc_train), end='', flush=True)
        print ('Loss(Test): {:.4f}, Accuracy(Test): {:.4f}'.format(loss_test, acc_test), flush=True)

    history = {'loss_train': losses_train,
               'acc_train': accs_train,
               'loss_test': losses_test,
               'acc_test': accs_test}

    return state, history


def main():
    print('\n')
    print('\n')
    print('\n')
    print('\n')

    (train_images, train_labels), (test_images, test_labels) = get_mnist_dataset()
    
    key, key1 = random.split(random.PRNGKey(0))
    variables = MyNet().init(key1, train_images[0:1])

    state = train_state.TrainState.create(
        apply_fn=MyNet().apply,
        params=variables['params'],
        tx=optax.adam(learning_rate=0.8)
    )

    state, history = fit(state, train_images, train_labels, test_images, test_labels, epochs=5)


if __name__ == '__main__':
    main()