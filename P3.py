import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

torch.manual_seed(47)

train = pd.read_csv("Train.csv")
X = train.drop(columns=["label", "attack_type", "num_outbound_cmds"])
y = train["label"]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=47, stratify=y)

scaler = joblib.load("scaler.joblib")
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)

#Converting it into tensor batches as PyTorch can only process them like this
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)

train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)

#going through all the layers in this order
model = nn.Sequential(
    nn.Linear(40, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print(model)

epochs = 100
loss_history = []

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor).squeeze(1)
    loss = loss_fn(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs} - loss: {loss.item():.4f}") #goes through all 100k+ records.
        #It measures every 100 records, predicts, clears and adjusts after


def show_results(model_name, y_true, y_pred):
    print("\n" + model_name)
    print("Accuracy :", round(accuracy_score(y_true, y_pred), 4))
    print("Precision:", round(precision_score(y_true, y_pred), 4))
    print("Recall   :", round(recall_score(y_true, y_pred), 4))
    print("F1-score :", round(f1_score(y_true, y_pred), 4))

model.eval()
with torch.no_grad():
    outputs = model(X_val_tensor).squeeze(1)
    probabilities = torch.sigmoid(outputs)
    nn_pred = (probabilities > 0.5).int().numpy()

show_results("Neural Network", y_val, nn_pred)

plt.plot(range(1, epochs + 1), loss_history)
plt.title("Neural network training loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
#plt.savefig("nn_loss_curve.png")
plt.show()

torch.save(model.state_dict(), "neural_network.pth")
#the neural network has more “power” than a Random Forest in theory,
#but it still came slightly behind.
#On table-style data like this, tree-based models often do
#as well as or better than neural networks, and they’re faster to train.