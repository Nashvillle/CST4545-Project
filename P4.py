import pandas as pd
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay

train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")

X_test = test.drop(columns=["label", "attack_type", "num_outbound_cmds"])
y_test = test["label"]

#--------------------------------------------------------------------------------------------
#importing scikit-learn models and predictig
lr = joblib.load("logistic_regression.joblib")
rf = joblib.load("random_forest.joblib")
dt = joblib.load("decision_tree.joblib")
scaler = joblib.load("scaler.joblib")

X_test_scaled = scaler.transform(X_test)

lr_pred = lr.predict(X_test_scaled)
rf_pred = rf.predict(X_test)
dt_pred = dt.predict(X_test)
#--------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------
#loading neural netwokr and predicting
model = nn.Sequential(
    nn.Linear(40, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)
model.load_state_dict(torch.load("neural_network.pth"))
model.eval()

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
with torch.no_grad():
    outputs = model(X_test_tensor).squeeze(1)
    probabilities = torch.sigmoid(outputs)
    nn_pred = (probabilities > 0.5).int().numpy()
#--------------------------------------------------------------------------------------------

#Comparing results and visualizing
def get_scores(y_true, y_pred):
    return [accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred),
            recall_score(y_true, y_pred),
            f1_score(y_true, y_pred)]

results = pd.DataFrame({
    "Logistic Regression": get_scores(y_test, lr_pred),
    "Decision Tree": get_scores(y_test, dt_pred),
    "Random Forest": get_scores(y_test, rf_pred),
    "Neural Network": get_scores(y_test, nn_pred)
}, index=["Accuracy", "Precision", "Recall", "F1-score"])

print("\nTest set results:")
print(results.T.round(4))

results.plot(kind="bar", figsize=(10, 6))
plt.title("Model comparison on the test set")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.savefig("model_comparison.png")
plt.show()

#Confusion Matrix to undertand the verity of results
all_preds = {"Logistic Regression": lr_pred, "Decision Tree": dt_pred,
             "Random Forest": rf_pred, "Neural Network": nn_pred}

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
i = 0
for name, pred in all_preds.items():
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["Normal", "Attack"],
        cmap="Blues", ax=axes[i], colorbar=False)
    axes[i].set_title(name)
    i = i + 1

plt.tight_layout()
plt.savefig("confusion_matrices.png")
plt.show()

#The best F1 score of the four models is Random Forest which caught the majority of the attacks
#Neural network and Descion Tree are almsot tied where it raised the lowest number of false alarms.

#confusion matrices show this clearly:
#about 4,600–4,900 missed attacks against only 234–639 false alarms,
#so models miss roughly 7–20 times more attacks than they wrongly flag.
#When they say “attack” they’re nearly always right,
# but over a third of real attacks get through.
# For an IDS, missed attacks are the more dangerous mistake.

#training is 47% attacks and test is 57%,
#and neptune makes up 70% of the training attacks