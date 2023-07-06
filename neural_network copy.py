import numpy as np
from utils import relu, relu, random_normal_distribution, drelu, into_batches, drelu, He
import functools
import foo_dataset


# ASSUMPTIONS:
# every layer has got relu as an activation function
# only neural layers are allowed
# every multi dimensional / non-vector input should be flatten  and normalised to E <0; 1>
class Neural_network:
    neural_layers = [] # assuming simple neural layer
    weights = [[]] # since input layer got no weights and for index (layers to weights) matching
    biases = [[]] # since input layer got no biases and for index (layers to biases) matching
    nabla_C = [[]]
    y = []
    accuracy = {"pos_sum": 0, "total_sum": 0}

    def __init__(self, input_size):
        self.neural_layers.append([])
        for i in range(input_size):
            self.neural_layers[0].append(0)
            
    def add_layer(self, n_neurons):
        current_last_layer_index = len(self.neural_layers) - 1
        current_last_layer = self.neural_layers[current_last_layer_index]
        # Initialize neurons with normal distribution
        self.neural_layers.append(np.full(n_neurons, random_normal_distribution()))
        self.biases.append(np.full(n_neurons, random_normal_distribution()))
        added_layer_index = current_last_layer_index + 1
        added_layer = self.neural_layers[added_layer_index]
        
        weights_between_layers_L_and_L_minus_1 = []
        for j, a_l in enumerate(added_layer):
            weights_of_n_neuron_L_to_L_minus_1_neurons = []
            for k, a_l_minus_1 in enumerate(current_last_layer):
                # Initialize weights with He initialization
                weights_of_n_neuron_L_to_L_minus_1_neurons.append(He(n_neurons))
            weights_between_layers_L_and_L_minus_1.append(weights_of_n_neuron_L_to_L_minus_1_neurons)
        self.weights.append(weights_between_layers_L_and_L_minus_1)
        self.nabla_C.append(weights_between_layers_L_and_L_minus_1)  # values do not matter its only for initializing gradient vector (here for coding purposes 3 dim)

    def __set_input_layer(self, neurons):
        if len(self.neural_layers) == 0:
            self.neural_layers.append(neurons)
        else:
            self.neural_layers[0] =  neurons

    def __accuracy(self, expected_j_index, predicted_j_index):
        if expected_j_index == predicted_j_index:
            self.accuracy["pos_sum"] += 1
        self.accuracy["total_sum"] += 1
        return self.accuracy["pos_sum"] / self.accuracy["total_sum"]


    # where label_value = [ label, value ]
    def __set_input(self, label_value_sets):
        self.input = label_value_sets
        input_layer = []
        #iterate through e.g. pic pixels / attrs
        for v in label_value_sets[0][1]:
            input_layer.append(v)
        self.__set_input_layer(input_layer)

    # j - neuron of layer L
    # k - neuron of layer L-1

    def __z_L_j(self, L, j):
        z_L_j = 0
        for k, _ in enumerate(self.neural_layers[L-1]):
            z_L_j += self.weights[L][j][k] * self.neural_layers[L-1][k] # + b[L][j]
        return z_L_j
    
    def __c_j(self, j):
        return 2 * (self.neural_layers[len(self.neural_layers) - 1][j] - self.y[j])

    def __dC_to_da_L_j(self, L, j):
        if(L == len(self.neural_layers) - 1):
            # Dla warstwy wyjściowej:
            return 2 * (self.neural_layers[L][j] - self.y[j])
        else:
            # Dla warstw ukrytych:
            sum = 0
            for neuron_idx, _ in enumerate(self.neural_layers[L+1]):
                sum += self.weights[L+1][neuron_idx][j] * drelu(self.__z_L_j(L+1, neuron_idx)) * self.__dC_to_da_L_j(L+1, neuron_idx)
            return sum
            
    def __dC_to_w_j_k_L(self, L, j, k):
        # print(L, j, k)
        return self.neural_layers[L-1][k] * drelu(self.__z_L_j(L, j)) * self.__dC_to_da_L_j(L, j)

    def __nabla_C(self):
        # nabla_C not one-dim vector but 3 (like wieghts)
        nabla_C = self.weights # only for initializing
        L = len(self.weights) - 1
        while L >= 1:
            for j, weights_L_j in enumerate(self.weights[L]):
                for k, _ in enumerate(weights_L_j):
                    # print(L, j, k)
                    nabla_C[L][j][k] = self.__dC_to_w_j_k_L(L, j, k)
            L -= 1
        return nabla_C
    
    def __avg_gradient(self, batch_input):
        gradients = []
        avg_gradient = self.nabla_C # only to set dimensions same as nablas C and same as weights
        sum_gradient = self.nabla_C # same as above + is only helper to contain sum  for later avg

        for input in batch_input:
            label = input[0]
            value = input[1]
            self.y = self.__y(label, self.classes)
            self.__set_input_layer(value)
            # self.__predict(value) needed if deleted from print
            print("Pred:", self.__predict(), "    y:", np.argmax(self.y), "    acc: ",  np.round(self.__accuracy(self.__predict(), np.argmax(self.y)), 3), " epoch:", self.epoch)
            self.nabla_C = self.__nabla_C() # recalculate nabla
            gradients.append(self.nabla_C)
            # print()

        for i_g, gradient_of_input in enumerate(gradients):
            for L, weights_L in enumerate(gradient_of_input):
                for j, wieghts_L_to_L_minus_1 in enumerate(weights_L):
                    for k, _ in enumerate(wieghts_L_to_L_minus_1):
                        sum_gradient[L][j][k] += gradient_of_input[L][j][k]
        
        for L, weights_L in enumerate(gradient_of_input):
            for j, wieghts_L_to_L_minus_1 in enumerate(weights_L):
                for k, _ in enumerate(wieghts_L_to_L_minus_1):
                    avg_gradient[L][j][k] = sum_gradient[L][j][k] / len(batch_input)
        
        return avg_gradient
    
    #calculate new weights based on gradient
    def __iterate_gradient_descent(self, batch):
        avg_gradient = self.__avg_gradient(batch)
        for L, weights_L in enumerate(avg_gradient):
            for j, wieghts_L_to_L_minus_1 in enumerate(weights_L):
                for k, _ in enumerate(wieghts_L_to_L_minus_1):
                    self.weights[L][j][k] -= (self.learning_rate * avg_gradient[L][j][k])

    def __y(self, current_label, classes):
        return list(map(lambda current_class: 1 if current_label == current_class else 0, classes))
    
    """
    returns index of prediction and sets neurons
    """
    def __predict(self):
        for L, layer in enumerate(self.neural_layers):
            if L == 0:
                continue
            for j, neuron_L_j in enumerate(layer):
                a_L_j = 0
                for k, neuron_L_minus_1_k in enumerate(self.neural_layers[L-1]):
                    a_L_j += self.weights[L][j][k] * neuron_L_minus_1_k
                self.neural_layers[L][j] = relu(a_L_j)
        return np.argmax(self.neural_layers[len(self.neural_layers) - 1])
    
    def __test(self, testing_set):
        sum_correct = 0
        sum_wrong = 0
        for test_set in testing_set:
            self.__predict(test_set[0])



    def train(self, training_set, testing_set, batch_size, epochs, classes, learning_rate, metric = 'accuracy'):
        self.classes = classes
        self.learning_rate = learning_rate
        for epoch in range(epochs):
            self.epoch = epoch
            batched_sets = into_batches(training_set, batch_size)
            for _, batched_set in enumerate(batched_sets):
                self.__iterate_gradient_descent(batched_set)


    def predict(self, input):
        self.__set_input_layer(input)
        self.__predict()
        return (self.neural_layers[len(self.neural_layers) - 1])



network = Neural_network(2)
network.add_layer(2)
network.add_layer(3)
network.add_layer(3)
network.add_layer(2)

network.train(training_set=foo_dataset.dataset, testing_set=[], batch_size=100, epochs=10, classes=foo_dataset.classes, learning_rate=0.01)