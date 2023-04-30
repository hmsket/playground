import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anm

DEAD = 0
ALIVE = 1

def generate_cells(nx, ny):
    rng = np.random.default_rng()
    cells = rng.integers(0, 2, (nx, ny))
    return cells

def count_alive_cell(cells, x, y, nx, ny):
    n = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            if x+i < 0 or x+i >= nx or y+j < 0 or y+j >= ny:
                continue
            if i == 0 and j == 0:
                continue
            if cells[x+i][y+j] == ALIVE:
                n = n+1
    return n

def update_cells(cells, nx, ny):
    tmp_cells = np.copy(cells)

    for i in range(nx):
        for j in range(ny):
            n = count_alive_cell(cells, i, j, nx, ny)
            if cells[i][j] == DEAD:
                if n == 3:
                    tmp_cells[i][j] = ALIVE
            else:
                if n not in [2, 3]:
                    tmp_cells[i][j] = DEAD

    cells = np.copy(tmp_cells)

    return cells

def main(nx=30, ny=30, N=50):
    fig = plt.figure()
    plt.xticks([])
    plt.yticks([])
    
    ims = []
    
    cells = generate_cells(nx, ny)
    im = plt.imshow(cells, cmap=plt.cm.gray_r, animated=True)
    ims.append([im])
    
    for _ in range(N):
        cells = update_cells(cells, nx, ny)
        im = plt.imshow(cells, cmap=plt.cm.gray_r, animated=True)
        ims.append([im])

    ani = anm.ArtistAnimation(fig, ims, interval=100)
    ani.save('out.gif', writer='imagemagick')
    
if __name__ == "__main__":
    main()