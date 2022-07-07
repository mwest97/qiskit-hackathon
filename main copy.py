import qiskit
from qiskit import Aer, QuantumCircuit
from qiskit.circuit import Parameter
import numpy as np
import torch
import torchvision
from torch import cat, no_grad, manual_seed
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from torch.nn import (
    Module,
    Conv2d,
    Linear,
    Dropout2d,
    NLLLoss,
    MaxPool2d,
    Flatten,
    Sequential,
    ReLU,
)
import torch.nn.functional as F
import matplotlib.pyplot as plt
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import CircuitQNN, TwoLayerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.opflow import AerPauliExpectation
from qiskit.utils import QuantumInstance, algorithm_globals

transform = transforms.Compose([
    transforms.CenterCrop(18),
    transforms.Resize(10),
    transforms.ToTensor()])
# Train Dataset
# -------------

# Set train shuffle seed (for reproducibility)
manual_seed(42)

batch_size = 1
n_samples = 100  # We will concentrate on the first 100 samples

# Use pre-defined torchvision function to load MNIST train data
X_train = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

# Filter out labels (originally 0-9), leaving only labels 0 and 1
idx = np.append(
    np.where(X_train.targets == 0)[0][:n_samples], np.where(X_train.targets == 1)[0][:n_samples]
)
X_train.data = X_train.data[idx]
X_train.targets = X_train.targets[idx]

# Define torch dataloader with filtered data
train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)

n_samples_show = 6

data_iter = iter(train_loader)
fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

while n_samples_show > 0:
    images, targets = data_iter.__next__()

    axes[n_samples_show - 1].imshow(images[0, 0].numpy().squeeze(), cmap="gray")
    axes[n_samples_show - 1].set_xticks([])
    axes[n_samples_show - 1].set_yticks([])
    axes[n_samples_show - 1].set_title("Labeled: {}".format(targets[0].item()))

    n_samples_show -= 1

# Test Dataset
# -------------

# Set test shuffle seed (for reproducibility)
# manual_seed(5)

n_samples = 50

# Use pre-defined torchvision function to load MNIST test data
X_test = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

# Filter out labels (originally 0-9), leaving only labels 0 and 1
idx = np.append(
    np.where(X_test.targets == 0)[0][:n_samples], np.where(X_test.targets == 1)[0][:n_samples]
)
X_test.data = X_test.data[idx]
X_test.targets = X_test.targets[idx]

# Define torch dataloader with filtered data
test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=True)

qi = QuantumInstance(Aer.get_backend("aer_simulator_statevector"))

# Define the encoding circuit
n_qbits = 3
n_layers = 3
encoding_circuit = QuantumCircuit(n_qbits)
encoding_circuit.h(range(n_qbits))

# Populate circuit
for n in range(n_qbits):
    encoding_circuit.rx(Parameter('e' + str(n)),n)
for n in range(n_qbits):
    encoding_circuit.cx(n,(n+1)%n_qbits)

parametrised_circuit = QuantumCircuit(n_qbits)
parametrised_circuit.h(range(n_qbits))

# Populate circuit
for l in range(n_layers):
    for n in range(n_qbits):
        parametrised_circuit.rx(Parameter('p' + str(n) + str(l)),n)
    for n in range(n_qbits):
        parametrised_circuit.cx(n,(n+1)%n_qbits)

# Define and create QNN
def create_qnn():
    feature_map = ZZFeatureMap(n_qbits)# encoding_circuit#ZZFeatureMap(2)
    ansatz = parametrised_circuit  # RealAmplitudes(n_qbits, reps=1)#parametrised_circuit#RealAmplitudes(2, reps=1)
    # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
    qnn = TwoLayerQNN(
        n_qbits,
        feature_map,
        ansatz,
        input_gradients=True,
        exp_val=AerPauliExpectation(),
        quantum_instance=qi,
    )
    return qnn

qnn = create_qnn()
#print(qnn.operator)

class Net(Module):
    def __init__(self, qnn):
        super().__init__()
        self.conv1 = Conv2d(1, 3, kernel_size=4)
        self.conv2 = Conv2d(3, 3, kernel_size=2)
        self.fc1 = Linear(27, n_qbits)
        self.qnn = TorchConnector(qnn)  # Apply torch connector, weights chosen

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.max_pool2d(x, 2)
        # print(x.shape)

        x = x.view(x.shape[0], -1)
        # print(x.shape)

        x = self.fc1(x)
        x = self.qnn(x)  # apply QNN
        return cat((x, 1 - x), -1)


class ClassicNet(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1, 6, kernel_size=4)
        self.conv2 = Conv2d(6, 3, kernel_size=2)
        self.fc1 = Linear(12, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))

        # print(x.shape)

        x = x.view(x.shape[0], -1)
        # print(x.shape)

        x = torch.nn.Softmax(dim=1)(self.fc1(x))
        # print(x)
        return x#cat((x, 1 - x), -1)


# model = Net(qnn)
classic_model = ClassicNet()
# Define model, optimizer, and loss function
# optimizer = optim.Adam(model.parameters(), lr=0.001)
classic_optimizer = optim.Adam(classic_model.parameters(), lr=0.001)

loss_func = NLLLoss()

# Start training
epochs = 5  # Set number of epochs
loss_list = []  # Store loss history
# model.train()  # Set model to training mode
#
# for epoch in range(epochs):
#     total_loss = []
#     for batch_idx, (data, target) in enumerate(train_loader):
#         optimizer.zero_grad(set_to_none=True)  # Initialize gradient
#         output = model(data)  # Forward pass
#         loss = loss_func(output, target)  # Calculate loss
#         loss.backward()  # Backward pass
#         optimizer.step()  # Optimize weights
#         total_loss.append(loss.item())  # Store loss
#     loss_list.append(sum(total_loss) / len(total_loss))
#     print("Training [{:.0f}%]\tLoss: {:.4f}".format(100.0 * (epoch + 1) / epochs, loss_list[-1]))
#
# model.eval()  # set model to evaluation mode
# with no_grad():
#
#     correct = 0
#     for batch_idx, (data, target) in enumerate(test_loader):
#         output = model(data)
#         if len(output.shape) == 1:
#             output = output.reshape(1, *output.shape)
#
#         pred = output.argmax(dim=1, keepdim=True)
#         correct += pred.eq(target.view_as(pred)).sum().item()
#
#         loss = loss_func(output, target)
#         total_loss.append(loss.item())
#
#     print(
#         "Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%".format(
#             sum(total_loss) / len(total_loss), correct / len(test_loader) / batch_size * 100
#         )
#     )
#
#     model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#     params = sum([np.prod(p.size()) for p in model_parameters])
# ----------------------------------------------------------------------------------------
# Start training
epochs = 5  # Set number of epochs
loss_list = []  # Store loss history
classic_model.train()  # Set model to training mode

for epoch in range(epochs):
    total_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        classic_optimizer.zero_grad(set_to_none=True)  # Initialize gradient
        output = classic_model(data)  # Forward pass
        loss = loss_func(output, target)  # Calculate loss
        loss.backward()  # Backward pass
        classic_optimizer.step()  # Optimize weights
        total_loss.append(loss.item())  # Store loss
    loss_list.append(sum(total_loss) / len(total_loss))
    print("Training [{:.0f}%]\tLoss: {:.4f}".format(100.0 * (epoch + 1) / epochs, loss_list[-1]))

classic_model.eval()  # set model to evaluation mode
with no_grad():

    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        output = classic_model(data)
        if len(output.shape) == 1:
            output = output.reshape(1, *output.shape)

        pred = output.argmax(dim=1, keepdim=True)
        print(output)
        correct += pred.eq(target.view_as(pred)).sum().item()

        loss = loss_func(output, target)
        total_loss.append(loss.item())

    print(
        "Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%".format(
            sum(total_loss) / len(total_loss), correct / len(test_loader) / batch_size * 100
        )
    )

    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])

