import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torch import save
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import transforms

device = "cpu"

# Get Data
transform = torchvision.transforms.Compose([transforms.ToTensor()
                                            , transforms.Normalize((0.5,), (0.5,))])

train_data = torchvision.datasets.MNIST('mnist_data', train=True, download=True, transform=transform)
val_data = torchvision.datasets.MNIST('mnist_data', train=False, download=True, transform=transform)

val_size = 2000
test_size = 8000

val_data, test_data = random_split(val_data, [val_size, test_size])

train_data = DataLoader(train_data, 64)
val_data = DataLoader(val_data, 64)
test_data = DataLoader(test_data, 64)


# Create Classifier Class
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels= 1, out_channels= 32 , kernel_size= (3,3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels= 64, kernel_size=(3,3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels= 128, kernel_size=(3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features= 61952, out_features= 10) # In order to avoid the change in resolution, do padding:
                                                            # use the formula ((W-F+2P) / S) + 1
        )

    def forward(self, x):
        return self.model(x)


# Create an instance of the Classifier
clf = NeuralNetwork().to(device)
loss_func = nn.CrossEntropyLoss()
optim = Adam(clf.parameters(), lr= 1e-3)


if __name__ == "__main__":
    # Train the model
    def train(epochs = 10, lr= 0.001, device = 'cpu'):
        for epoch in range(epochs):
            for batch_tr in train_data:
                x, y = batch_tr
                x, y = x.to(device), y.to(device)
                pred_tr = clf(x)
                loss_tr = loss_func(pred_tr, y)
                optim.zero_grad()
                loss_tr.backward()
                optim.step()

            accuracy = validate(clf, val_data)

            accuracy_lst[epoch] = accuracy
            tr_losslst[epoch] = loss_tr

            print(f"epoch: {epoch + 1}, loss_tr: {loss_tr}")
            print(f"epoch: {epoch + 1}, accuracy: {accuracy} %")

        tr_losslst_np = tr_losslst.detach().numpy()
        accuracy_lst_np = accuracy_lst.detach().numpy()

        plt.plot(range(epochs), tr_losslst_np, label="tr_loss")
        plt.ylabel("training loss")
        plt.xlabel("Epoch")
        plt.show()

        plt.plot(range(epochs), accuracy_lst_np, label="Accuracy")
        plt.ylabel("Accuracy (%)")
        plt.xlabel("Epoch")
        plt.show()


    def validate(model, val_data):
        total = 0
        correct = 0

        with torch.no_grad():
            for batch_val in val_data:
                x, y = batch_val
                x, y = x.to(device), y.to(device)
                pred = model(x)
                total += pred.size(0)
                for i in range(pred.size(0)):
                    if pred[i].argmax().item() == y[i].item():
                        correct += 1
        return (correct / total) * 100.


    # Train and Check the accuracy

    epochs = 10
    tr_losslst = torch.zeros(epochs)
    accuracy_lst = torch.zeros(epochs)

    train_model = train(epochs,lr= 0.001)


    with open("mnist_model.pt", 'wb') as file:
        save(clf.state_dict(), file)

