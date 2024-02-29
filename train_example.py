# ================================================================ #
#                       LSTM Neural Networks                       #
# ================================================================ #
import torch
import pandas as pd
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms

import tqdm

batch_size = 100  # Размер пакета (batch size) - количество образцов, обрабатываемых за одну итерацию обучения.
num_epochs = 10  # Количество эпох - количество полных проходов через набор данных во время обучения.
learning_rate = 0.1  # Скорость обучения (learning rate) - определяет, насколько быстро модель обновляет свои веса.

input_dim = 3  # Размерность входных данных - количество признаков во входных данных.
hidden_dim = 2  # Размерность скрытого слоя - количество нейронов в скрытом слое.
sequence_dim = 288  # Размерность последовательности - количество временных шагов во входных данных.
layer_dim = 256  # Количество слоев LSTM - количество LSTM слоев в модели.
output_dim = 2  # Размерность выходных данных - количество классов, которые модель может предсказывать.

# Конфигурация устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Определение устройства для обучения модели (GPU или CPU).

# ================================================================ #
#                        Data Loading Process                      #
# ================================================================ #

# Dataset

class CryptoDataset(torch.utils.data.Dataset):
    def __init__(self, filename, date_field="date"):
        self.data = pd.read_json(filename)
        self.data[date_field] = pd.to_datetime(self.data[date_field], unit="ms")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        date, rate, volume, cap = self.data.iloc[index]
        return date, rate, volume, cap

train_dataset = CryptoDataset("data.json")

# FIXME: Split pandas dataframe into train and test
test_dataset = CryptoDataset("data.json")

# Split pandas dataframe into train and test


# Data Loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# ================================================================ #
#                       Create Model Class                         #
# ================================================================ #

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out


model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# ================================================================ #
#                           Train and Test                         #
# ================================================================ #

# Train the model
iter = 0
print('TRAINING STARTED.\n')
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, sequence_dim, input_dim).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter += 1
        if iter % 500 == 0:
            # Calculate Loss
            print(f'Epoch: {epoch + 1}/{num_epochs}\t Iteration: {iter}\t Loss: {loss.item():.2f}')

# Test the model
model.eval()
print('\nCALCULATING ACCURACY...\n')
with torch.no_grad():
    correct = 0
    total = 0
    progress = tqdm.tqdm(test_loader, total=len(test_loader))
    # Iterate through test dataset
    for images, labels in progress:
        images = images.view(-1, sequence_dim, input_dim).to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        # Total number of labels
        total += labels.size(0)

        # Total correct predictions
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    # Print Accuracy
    print(f'Accuracy: {accuracy}')