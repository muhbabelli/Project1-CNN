import mnist_classifier as cl
import torch

mydevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = cl.NeuralNetwork()
model.load_state_dict(torch.load("mnist_model.pt"))
model.eval() # Set the model to evaluation mode (important for models with dropout or batch normalization)

# Assuming you have a test dataset named 'test_data'
test_loader = cl.test_data


print("Starting test!")
correct = 0
total = 0
count = 1
with torch.no_grad():
    for batch_test in test_loader:
        x, y = batch_test
        x, y = x.to(mydevice), y.to(mydevice)
        pred = model(x)
        total += pred.size(0)
        for i in range(pred.size(0)):
            if pred[i].argmax().item() == y[i].item():
                correct += 1
        print(f"Batch: {count} Done -> Accuracy: {(correct/total)*100:.4f}% ")
        count += 1

accuracy = 100 * correct / total
print("----------------")
print(f"Test Accuracy: {accuracy:.4f}%")


