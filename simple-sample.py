from tinygrad import Tensor
from tinygrad.nn.optim import SGD
from tinygrad import Device
from tinygrad import nn
from tinygrad.nn.datasets import mnist
from tinygrad.helpers import Timing

print(f"Device: {Device.DEFAULT}")

class TinyMNISTNet:
    def __init__(self):
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3))
        self.fc1 = nn.Linear(1600, 250)
        self.fc2 = nn.Linear(250, 10)

    def __call__(self, x):
        x = self.conv1(x).max_pool2d(2).relu()
        x = self.conv2(x).max_pool2d(2).relu()
        x = x.flatten(1)
        x = self.fc1(x).relu().dropout(0.5)
        x = self.fc2(x)
        return x.log_softmax()

class TinyMNISTNetInvestigation:
    def __init__(self):
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5,5))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(7,7))
        self.fc1 = nn.Linear(3200, 250)
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

class TinyMNISTNRNBCNet:
    def __init__(self):
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3))
        self.bn1 = nn.BatchNorm(32)
        self.bn1.weight.requires_grad = False
        self.conv2 = nn.Conv2d(32, 48, kernel_size=(3,3))
        self.bn2 = nn.BatchNorm(48)
        self.bn2.weight.requires_grad = False
        self.conv3 = nn.Conv2d(48, 64, kernel_size=(3,3))
        self.bn3 = nn.BatchNorm2d(64)
        # self.bn3.weight.requires_grad = False
        # self.conv4 = nn.Conv2d(64, 80, kernel_size=(3,3))
        # self.bn4 = nn.BatchNorm2d(80)
        # self.bn4.weight.requires_grad = False
        # self.conv5 = nn.Conv2d(80, 96, kernel_size=(3,3))
        # self.bn5 = nn.BatchNorm2d(96)
        # self.bn5.weight.requires_grad = False
        # self.conv6 = nn.Conv2d(96, 112, kernel_size=(3,3))
        # self.bn6 = nn.BatchNorm2d(112)
        # self.bn6.weight.requires_grad = False
        # self.conv7 = nn.Conv2d(112, 128, kernel_size=(3,3))
        # self.bn7 = nn.BatchNorm2d(128)
        # self.bn7.weight.requires_grad = False
        # self.conv8 = nn.Conv2d(128, 144, kernel_size=(3,3))
        # self.bn8 = nn.BatchNorm2d(144)
        # self.bn8.weight.requires_grad = False
        # self.conv9 = nn.Conv2d(144, 160, kernel_size=(3,3))
        # self.bn9 = nn.BatchNorm2d(160)
        # self.bn9.weight.requires_grad = False
        self.fc1 = nn.Linear(30976, 1250)
        self.fc2 = nn.Linear(1250, 250)
        self.fc3 = nn.Linear(250, 10)
        # self.fc3 = nn.Linear(16000, 250)


    def __call__(self, x):
        x = self.bn1(self.conv1(x)).relu()
        x = self.bn2(self.conv2(x)).relu()
        x = self.bn3(self.conv3(x)).relu()
        # y = x
        x = x.softmax()
        # y = self.bn4(self.conv4(y)).relu()
        # y = self.bn5(self.conv5(y)).relu()
        # y = self.bn6(self.conv6(y)).relu()
        # z = y
        # y = y.softmax()
        # z = self.bn7(self.conv7(z)).relu()
        # z = self.bn8(self.conv8(z)).relu()
        # z = self.bn9(self.conv9(z)).relu()
        # z = z.relu()
        x = x.flatten(1)
        x = self.fc1(x).relu().dropout(0.5)
        # y = y.flatten(1)
        # y = self.fc2(y).relu().dropout(0.5)
        # z = z.flatten(1)
        # z = self.fc3(z).relu().dropout(0.5)
        # x = x + y + z  # INCOMPLETE
        x = self.fc2(x).relu()
        x = self.fc3(x).relu()
        return x.log_softmax()

net = TinyMNISTNet()
# net = TinyMNISTNetInvestigation()
# net = TinyMNISTNRNBCNet()

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
