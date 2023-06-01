# https://qiita.com/kuroitu/items/35d7b5a4bde470f69570

import numpy as np

input_size = (3,3)

image = np.arange(1,10)
image = image.reshape(input_size)

kernel_size = (2,2)
output_size = (input_size[0]-kernel_size[0]+1, input_size[1]-kernel_size[1]+1)

col = np.empty((kernel_size[0], kernel_size[1], output_size[0], output_size[1]))

for i in range(kernel_size[0]):
    for j in range(kernel_size[1]):
        col[i, j, :, :] = image[i:i+2, j:j+2]

col = col.reshape(4,4)

print(col)