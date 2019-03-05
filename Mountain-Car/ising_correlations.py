import numpy as np
import numpy.random as rnd


def random_positions(Lx, Ly, N):
    pos = np.zeros((N, 3))
    pos[:, 0] = np.random.randint(0, Lx, N)
    pos[:, 1] = np.random.randint(0, Ly, N)
    pos[:, 2] = np.random.randint(0, 2, N) * 2 - 1
    pos = remove_repeated_positions(pos)
    return pos


def move_one_position(I, mode='all'):

    pos = I.pos
    if mode == 'all':
        i = np.random.randint(I.size)
    elif mode == 'sensors':
        i = np.random.randint(0, I.Ssize)
    elif mode == 'motors':
        i = np.random.randint(I.Msize) + I.size - I.Msize
    elif mode == 'hidden':
        i = np.random.randint(I.Ssize, I.size - I.Msize)

    r = rnd.rand()
    if (r < (1 / 3)):
        pos[i, 0] += np.random.randint(2) * 2 - 1
    elif(r < (2 / 3)):
        pos[i, 1] += np.random.randint(2) * 2 - 1
    else:
        pos[i, 2] *= -1
    pos = remove_repeated_positions(pos)
    return pos


def remove_repeated_positions(pos):
    size = pos.shape[0]
    values, index, counts = np.unique(
        pos[:, 0:2], axis=0, return_index=True, return_counts=True)
    while max(counts) > 1:
        for i in range(len(counts)):
            if counts[i] > 1:
                if np.random.rand() > 0.5:
                    pos[index[i], 0] += np.random.randint(2) * 2 - 1
                else:
                    pos[index[i], 1] += np.random.randint(2) * 2 - 1
        values, index, counts = np.unique(
            pos[:, 0:2], axis=0, return_index=True, return_counts=True)
    return pos


def random_means(N):
    return np.random.rand(N) * 2 - 1
#  return np.random.triangular(-1, 0, 1, N)


def ising_correlations(pos, m):
    size = pos.shape[0]
    C = np.zeros((size, size))
    for i in range(size):
        #    C[i,i] = 1 - m[i]**2
        for j in range(i + 1, size):
            r = np.abs(pos[i, 0] - pos[j, 0]) + np.abs(pos[i, 1] - pos[j, 1])
            sign = pos[i, 2] * pos[j, 2]
            C[i, j] = sign * 0.9 * r**-0.25
#      C[j,i] = C[i,j]
    return C

# N=5
# Lx=100
# Ly=100
#pos=random_positions(Lx, Ly, N)
# m=random_means(N)
# print(ising_correlations(pos,m))
