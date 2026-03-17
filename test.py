import torch
import numpy as np
import requests, gzip, hashlib, os
import cv2

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


test_data = fetch("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1,28,28))
test_targets = fetch("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz")[8:]


network = Net()

network.load_state_dict(torch.load("Models/06_pytorch_introduction/model.pt"))

network.eval()


for test_image,test_target in zip(test_data,test_targets):

    inference_image = torch.from_numpy(test_image).float()/255.0

    inference_image = inference_image.unsqueeze(0).unsqueeze(0)

    output = network(inference_image)

    pred = output.argmax(dim=1)

    prediction = str(pred.item())

    test_image = cv2.resize(test_image,(400,400))

    cv2.imshow(prediction,test_image)

    key = cv2.waitKey(0)

    if key == ord("q"):
        break

cv2.destroyAllWindows()