# MNIST Digit Classification using PyTorch

This project implements a Convolutional Neural Network (CNN) for handwritten digit classification using the MNIST dataset.

---

## 📌 Model Architecture

* Conv2D → ReLU → MaxPool
* Conv2D → ReLU → MaxPool
* Fully Connected Layer
* Output Layer (10 classes)

---

## 📁 Project Structure

```
model.py          → CNN architecture  
train.py          → training pipeline  
test.py           → inference  
requirements.txt  → dependencies  
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## 🚀 Training

```bash
python train.py
```

---

## 🔍 Testing

```bash
python test.py
```

---

## 📊 Results

* Accuracy: ~98%
* Dataset: MNIST
* Framework: PyTorch

---

## 📦 Dataset

The MNIST dataset is automatically downloaded during training.

---

## 🛠️ Tech Stack

* Python
* PyTorch
* NumPy
* OpenCV

---

