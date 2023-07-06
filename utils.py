import math
import numpy as np
import random

def random_normal_distribution():
    return np.random.normal(0.5, 0.3, 1)

def sigmoid(x):
    x = np.clip(x, -500, 500)  # Ogranicz wartości do pewnego zakresu przed zastosowaniem funkcji exp
    return 1. / (1. + np.exp(-x))

def dsigmoid(x):
    return sigmoid(x) * (1. - sigmoid(x))

def relu(x):
    return max(0., x)

def drelu(x):
    return 1. if x > 0. else 0.

def He(n_neurons):
    return np.random.randn() * np.sqrt(2 / n_neurons)

def into_batches(arr, batch_size):
    random.shuffle(arr)
    # print('Batching: ', len(arr), arr, batch_size) # swietnie pomieszal xDD
    return np.array_split(arr, len(arr)/batch_size)





