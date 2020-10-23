# To start, import PennyLane, NumPy, and PyTorch for the optimization:

import pennylane as qml
import numpy as np
import torch
from torch.autograd import Variable

np.random.seed(42)

# generate a three-dimensional random vector by sampling
# each entry from a standard normal distribution
v = np.random.normal(0, 1, 3)

# purity of the target state
purity = 0.66

# create a random Bloch vector with the specified purity
bloch_v = np.sqrt(2 * purity - 1) * v / np.sqrt(np.sum(v ** 2))

# array of Pauli matrices (will be useful later)
Paulis = np.zeros((3, 2, 2), dtype=complex)
Paulis[0] = [[0, 1], [1, 0]]
Paulis[1] = [[0, -1j], [1j, 0]]
Paulis[2] = [[1, 0], [0, -1]]

##############################################################################
# Unitary operations map pure states to pure states. So how can we prepare
# mixed states using unitary circuits? The trick is to introduce
# additional qubits and perform a unitary transformation on this larger
# system. By "tracing out" the ancilla qubits, we can prepare mixed states
# in the target register. In this example, we introduce two additional
# qubits, which suffices to prepare arbitrary states.
#
# The ansatz circuit is composed of repeated layers, each of which
# consists of single-qubit rotations along the :math:`x, y,` and :math:`z`
# axes, followed by three CNOT gates entangling all qubits. Initial gate
# parameters are chosen at random from a normal distribution. Importantly,
# when declaring the layer function, we introduce an input parameter
# :math:`j`, which allows us to later call each layer individually.

# number of qubits in the circuit
nr_qubits = 3
# number of layers in the circuit
nr_layers = 2

# randomly initialize parameters from a normal distribution
params = np.random.normal(0, np.pi, (nr_qubits, nr_layers, 3))
params = Variable(torch.tensor(params), requires_grad=True)

# a layer of the circuit ansatz
def layer(params, j):
    for i in range(nr_qubits):
        qml.RX(params[i, j, 0], wires=i)
        qml.RY(params[i, j, 1], wires=i)
        qml.RZ(params[i, j, 2], wires=i)

    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[0, 2])
    qml.CNOT(wires=[1, 2])


##############################################################################
# Here, we use the ``default.qubit`` device to perform the optimization, but this can be changed to
# any other supported device.

dev = qml.device("default.qubit", wires=3)

##############################################################################
# When defining the QNode, we introduce as input a Hermitian operator
# :math:`A` that specifies the expectation value being evaluated. This
# choice later allows us to easily evaluate several expectation values
# without having to define a new QNode each time.
#
# Since we will be optimizing using PyTorch, we configure the QNode
# to use the PyTorch interface:


@qml.qnode(dev, interface="torch")
def circuit(params, A=None):

    # repeatedly apply each layer in the circuit
    for j in range(nr_layers):
        layer(params, j)

    # returns the expectation of the input matrix A on the first qubit
    return qml.expval(qml.Hermitian(A, wires=0))


##############################################################################
# Our goal is to prepare a state with the same Bloch vector as the target
# state. Therefore, we define a simple cost function
# Finally, we compare the Bloch vectors of the target and output state.

# cost function
def cost_fn(params):
    cost = 0
    for k in range(3):
        cost += torch.abs(circuit(params, A=Paulis[k]) - bloch_v[k])

    return cost


# set up the optimizer
opt = torch.optim.Adam([params], lr=0.1)

# number of steps in the optimization routine
steps = 200

# the final stage of optimization isn't always the best, so keep track of
# the best parameters along the way
best_cost = cost_fn(params)
best_params = np.zeros((nr_qubits, nr_layers, 3))

print("Cost after 0 steps is {:.4f}".format(cost_fn(params)))

# optimization begins
for n in range(steps):
    opt.zero_grad()
    loss = cost_fn(params)
    loss.backward()
    opt.step()

    # keeps track of best parameters
    if loss < best_cost:
        best_params = params

    # Keep track of progress every 10 steps
    if n % 10 == 9 or n == steps - 1:
        print("Cost after {} steps is {:.4f}".format(n + 1, loss))

# calculate the Bloch vector of the output state
output_bloch_v = np.zeros(3)
for l in range(3):
    output_bloch_v[l] = circuit(best_params, A=Paulis[l])

# print results
print("Target Bloch vector = ", bloch_v)
print("Output Bloch vector = ", output_bloch_v)
