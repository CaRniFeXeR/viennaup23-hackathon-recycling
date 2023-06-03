import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from extract_features import get_features
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np
import json
import os


def load_data(semi=True, root_dir='data/'):
    if semi:
        features = get_features()

        features_torch = []
        class_ids_torch = []

        # Extract features and make tensors
        for feature in features:
            feature_tensor = feature[0].detach().cpu()
            label = feature[1]
            class_id = feature[2]

            features_torch.append(feature_tensor)
            class_ids_torch.append(torch.as_tensor(class_id))

        # Stack lists:
        X_data = torch.stack(features_torch)
        y_data = torch.stack(class_ids_torch)
    else:
        data_json = json.load(open(root_dir + 'labels.json'))

        features_path = os.path.join(root_dir, 'images')

        features_torch = []
        class_ids_torch = []
        for file in data_json:
            print(file, data_json[file])
            file = file.replace('.jpg', '.npy')
            features = np.load(os.path.join(features_path, file))

            features_torch.append(torch.as_tensor(features))
            class_ids_torch.append(torch.as_tensor(data_json[file]))

        # Stack lists:
        X_data = torch.stack(features_torch)
        y_data = torch.stack(class_ids_torch)

    return X_data, y_data, class_ids_torch


class FloKo(nn.Module):
    def __init__(self, class_ids):
        super(FloKo, self).__init__()
        self.fc = nn.Linear(768, 50)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(50, 2)
        self.fc4 = nn.Linear(2, torch.stack(class_ids).max() + 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        intermediate = self.fc3(x)
        x = self.fc4(intermediate)
        x = self.sigmoid(x)

        return x, intermediate


def train_model(model, X_train, y_train, num_epochs=100, batch_size=10, device='cuda'):
    model.to(device)

    # Split data into train and test
    X_test = X_train[int(len(X_train) * 0.8):]
    y_test = y_train[int(len(y_train) * 0.8):]

    # Create a TensorDataset
    dataset_train = torch.utils.data.TensorDataset(X_train, y_train)
    # Create a TensorDataset
    dataset_test = torch.utils.data.TensorDataset(X_test, y_test)
    # Create a DataLoader
    dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    losses = []
    for epoch in trange(num_epochs):
        total_loss = []
        for inputs, targets in dataloader:
            # Zero the gradients
            optimizer.zero_grad()

            inputs = inputs.to("cuda")
            targets = targets.to("cuda")

            # Encode targets one-hot
            targets = torch.nn.functional.one_hot(targets, num_classes=3).float()

            # Forward pass
            output, intermediate = model(inputs)

            # Compute the loss
            loss = loss_function(output, target=targets)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()
            total_loss.append(loss.item())
        losses.append(np.mean(total_loss))

    return losses


if __name__ == '__main__':
    X_data, y_data, class_ids = load_data(True)
    model = FloKo(class_ids=class_ids)

    print(summary(model, (768,)))

    losses = train_model(model, X_data[:int(len(X_data) * 0.8)], y_data[:int(len(y_data) * 0.8)])
