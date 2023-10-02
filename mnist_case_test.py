from io import BytesIO

import mnist_classifier as cl
import numpy as np
import requests
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

device = "cpu"

def inference(path, model, device):
    T = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    r = requests.get(path)
    with BytesIO(r.content) as f:
        img = Image.open(f).convert(mode="L")
        img = img.resize((28, 28))
        x = (255 - np.expand_dims(np.array(img), -1)) / 255.
    with torch.no_grad():
        pred = model(torch.unsqueeze(T(x), axis=0).float().to(device))
        return F.softmax(pred, dim=-1).numpy()

model = cl.NeuralNetwork()
model.load_state_dict(torch.load("mnist_model.pt"))
model.eval()

with open("mnist_model.pt", 'rb') as f:
    path = ("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRaZZZF39S3WJQa-M3nxD0tbn-fiH3p51Nt4Q&usqp=CAU")
    pred = inference(path, model, device='cpu')
    pred_idx = np.argmax(pred)
    print(f"Predicted: {pred_idx}, Prob: {pred[0][pred_idx] * 100} %")

# Fails:
# https://i.stack.imgur.com/RdEpj.png
# https://digilent.com/blog/wp-content/uploads/2019/10/AI-article-1.png
# https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQfGtI01uzTRsnDhDCPF6d25HpGyal_JAJq4g&usqp=CAU


# Success:
# https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTWmgVCR-9bNRLhTNrjFmEQUHyBZhM5PoRLdQ&usqp=CAU
# https://d3i71xaburhd42.cloudfront.net/7b14ccbaf08683e3e284d9bfded0712dab8f86ba/3-Figure3-1.png
# https://i.stack.imgur.com/FK0FB.png
# https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRaZZZF39S3WJQa-M3nxD0tbn-fiH3p51Nt4Q&usqp=CAU



# Cannot Identify:
# https://www.researchgate.net/publication/287853768/figure/fig4/AS:667034937462784@1536044930113/Handwritten-digits-from-the-MNIST-data-set-5-For-practical-machine-learning-tasks.jpg


