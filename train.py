import os
import numpy as np
from tqdm import tqdm
import requests, gzip, hashlib

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from model import Net

path = "Datasets/mnist"

def fetch(url):

    if not os.path.exists(path):
        os.makedirs(path)

    fp = os.path.join(path, hashlib.md5(url.encode("utf-8")).hexdigest())

    if os.path.isfile(fp):

        with open(fp, "rb") as f:
            data = f.read()

    else:

        with open(fp, "wb") as f:
            data = requests.get(url).content
            f.write(data)

    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()


train_data = fetch("https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1,28,28))
train_targets = fetch("https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz")[8:]


n_epochs = 5
batch_size = 64
learning_rate = 0.001


train_data = np.expand_dims(train_data, axis=1) / 255.0
test_data = np.expand_dims(test_data, axis=1) / 255.0


train_batches = [np.array(train_data[i:i+batch_size]) for i in range(0,len(train_data),batch_size)]
train_target_batches = [np.array(train_targets[i:i+batch_size]) for i in range(0,len(train_targets),batch_size)]

test_batches = [np.array(test_data[i:i+batch_size]) for i in range(0,len(test_data),batch_size)]
test_target_batches = [np.array(test_targets[i:i+batch_size]) for i in range(0,len(test_targets),batch_size)]


network = Net()

summary(network,(1,28,28))


optimizer = optim.Adam(network.parameters(),lr=learning_rate)

loss_function = nn.CrossEntropyLoss()


def train(epoch):

    network.train()

    loss_sum = 0

    train_pbar = tqdm(zip(train_batches,train_target_batches),total=len(train_batches))

    for index,(data,target) in enumerate(train_pbar,start=1):

        data = torch.from_numpy(data).float()
        target = torch.from_numpy(target).long()

        optimizer.zero_grad()

        output = network(data)

        loss = loss_function(output,target)

        loss.backward()

        optimizer.step()

        loss_sum += loss.item()

        train_pbar.set_description(f"Epoch {epoch}, loss: {loss_sum/index:.4f}")


def test(epoch):

    network.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        val_pbar = tqdm(zip(test_batches,test_target_batches),total=len(test_batches))

        for data,target in val_pbar:

            data = torch.from_numpy(data).float()
            target = torch.from_numpy(target).long()

            output = network(data)

            pred = output.argmax(dim=1)

            correct += (pred==target).sum().item()

            total += target.size(0)

            val_pbar.set_description(f"Accuracy: {correct/total:.4f}")


for epoch in range(1,n_epochs+1):

    train(epoch)

    test(epoch)


output_path = "Models/06_pytorch_introduction"

os.makedirs(output_path,exist_ok=True)

torch.save(network.state_dict(),os.path.join(output_path,"model.pt"))

print("Model Saved")