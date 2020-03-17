import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
# make experiments reproducible
torch.manual_seed(1)

device = "cpu"

# define the simulation
def simulation(x):
    """ run simulation with input x, produce lable y"""
    y = ((x-target)**2).sum(1).view(-1,1)
    return y

# define the target
target = torch.tensor([0.5, -1.88], device=device)
print("target", target)

# start from random x
x = torch.randn([100,2], device=device, requires_grad=True)

# define a neural network (Layers: 1->32->32->1)
class Net(nn.Module):
    def __init__(self):
        num_neurons = 32
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, num_neurons)
        self.fc2 = nn.Linear(num_neurons, num_neurons)
        self.fc3 = nn.Linear(num_neurons, 1)

        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net().to(device=device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

mutation_rate = 0.5

history = {}
history1 = {}
for j in range(5):
    history[j] = []
    history1[j] = []

# Training loop
for i in range(300):
    # Step 1. Do simulation
    y = simulation(x)

    # Step 2. Use the result of simulation to train the model
    for step in range(10):
        y_hat = model(x)
        loss = nn.MSELoss()(y_hat, y)
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

    # Step 3. Use the model to tune the x
    for step in range(1):
        y_hat = model(x)
        loss_x = y_hat.mean()
        loss_x.backward(retain_graph=True)
        with torch.no_grad():
            x -= mutation_rate * x.grad
        x.grad.zero_()
        for j in range(5):
            history[j].append(x.detach().numpy()[j][0])
            history1[j].append(x.detach().numpy()[j][1])

# Plot the results
colors = ["red", "blue", "green", "orange", "purple"]
xx = range(len(history[0]))
yy = [0.5] * len(history[0])
yy1 = [-1.88] * len(history1[0])
plt.plot(xx,yy,label="truth for x[0]", color="#FF3322", linewidth=2)
plt.plot(xx,yy1,label="truth for x[1]", color="#FF3322", linewidth=2)
for j in range(5):
    plt.plot(xx, history[j],  c=colors[j],  linewidth=1)
    plt.plot(xx, history1[j],  c=colors[j],  linewidth=1)
plt.legend()
plt.ylabel("x[0] and x[1]")
plt.xlabel("epoch")
plt.show()

