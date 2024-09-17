from tinygrad import Tensor
from tinygrad.nn.optim import SGD
from tinygrad import Device
from tinygrad import nn
from tinygrad.nn.datasets import mnist
from tinygrad.helpers import Timing

print("Device: {Device.DEFAULT}")

class TinyMNISTNet:
    def __init__(self):
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3))
        self.fc1 = nn.Linear(4096, 250)
        self.fc2 = nn.Linear(250, 10)

    def __call__(self, x):
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = x.max_pool2d((3,3))
        x = x.flatten(1)
        x = x.dropout(0.5)
        x = self.fc1(x).sigmoid()
        x = self.fc2(x)
        return x.softmax()

class TinyMNISTNetInvestigation:
    def __init__(self):
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3))
        self.fc1 = nn.Linear(6272, 250)
        self.fc2 = nn.Linear(250, 10)

    def __call__(self, x):
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = self.conv3(x).relu()
        x = x.max_pool2d((3,3))
        x = x.flatten(1)
        x = x.dropout(0.5)
        x = self.fc1(x).sigmoid()
        x = self.fc2(x)
        return x.softmax()

# net = TinyMNISTNet()
net = TinyMNISTNetInvestigation()

optim = nn.optim.Adam(nn.state.get_parameters(net))

X_train, Y_train, X_test, Y_test = mnist()

print(f"Number of samples to train {len(X_train)}")

batch_size = 256
def step():
    Tensor.training = True  # makes dropout work
    samples = Tensor.randint(batch_size, high=X_train.shape[0])
    X, Y = X_train[samples], Y_train[samples]
    optim.zero_grad()
    Y_calc = net(X)
    loss = Y_calc.sparse_categorical_crossentropy(Y).backward()
    optim.step()
    pred = Y_calc.argmax(axis=1)
    acc = (pred == Y).mean()
    print(f"Loss: {loss.numpy()} | Accuracy: {acc.numpy()}")

    return loss

for step_num in range(200):
    loss = step()
    
    if step_num%10 == 0:
        Tensor.training = False
        acc = (net(X_test).argmax(axis=1) == Y_test).mean().item()
        print(f"loss {loss.item():.2f}, acc {acc*100.:.2f}%")
