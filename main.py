import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])

# data output
training_outputs = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)
synaptic_weights = 2 * np.random.random((3, 1)) - 1

print('\nsynaptic_weights: ')
print(synaptic_weights)

for iteration in range(1):
    input_layer = training_inputs
    print('\ninput_layer: ')
    print(input_layer)

    outputs = sigmoid(np.dot(input_layer, synaptic_weights))
    print('\nOutputs: ')
    print(outputs)

    # update with error
    error = training_outputs - outputs
    print('\nerror: ')
    print(error)

    adjustments = error * sigmoid_derivative(outputs)
    print('\nadjustments: ')
    print(adjustments)

    synaptic_weights += np.dot(input_layer.T, adjustments)
    print('\nsynaptic weights: ')
    print(synaptic_weights)


#print('Synaptic weight after training')
# print(synaptic_weights)

#print('Output training')
# print(outputs)
