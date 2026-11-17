print('Import the provided training and testing datasets into Python')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")

print('Report the number of rows and columns in both datasets')
print("Train shape:", train.shape)
print("Test shape:", test.shape)

print('Descriptive statistics')
print(train.describe())
print('----------------------------------------------------------------------------------')

print('Calculate the proportion of normal and attack traffic within the training dataset.')
#Of all the traffic in the training set, what percentage is normal and what percentage is attacks?
train_percent =train["label"].value_counts(normalize=True) * 100
# The answer is about 53.5% normal and 46.5% attacks.
test_percent = test["label"].value_counts(normalize=True) * 100
print("Training set (%):")
print(train_percent)
print("Testing set (%):")
print(test_percent)
print('----------------------------------------------------------------------------------')

print("Produce bar charts comparing the number of normal and attack records in the training and testing datasets.")
percent = pd.DataFrame({"Train": train_percent, "Test": test_percent})
percent.index = ["Normal", "Attack"]
percent.plot(kind="bar")
plt.title("Normal vs Attack records (%)")
plt.ylabel("Percentage of records (%)")
plt.xticks(rotation=0)
plt.savefig("bar_chart_percent.png")
plt.show()
#Testing model shows more attack vectors as compared to Training model
print('----------------------------------------------------------------------------------')

print('Generate a correlation heatmap to examine relationships between the features in the training dataset')
numbers_only = train.drop(columns=["attack_type", "num_outbound_cmds"])

plt.figure(figsize=(16, 13))
sns.heatmap(numbers_only.corr(), cmap="coolwarm")
plt.title("Correlation heatmap of training set features")
plt.tight_layout()
plt.savefig("heatmap.png")
plt.show()
print('----------------------------------------------------------------------------------')


# attacks label = 1
attacks = train[train["label"] == 1]

# Count each attack type and turn the counts into percentages
attack_percent = attacks["attack_type"].value_counts(normalize=True) * 100
print(attack_percent)

# Bar chart of the attack types
attack_percent.plot(kind="bar", figsize=(12, 6))
plt.title("Attack types in the training data (%)")
plt.xlabel("Attack type")
plt.ylabel("Percentage of attacks (%)")
plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig("attack_types.png")
plt.show()
#neptune is about 70% of all attacks followed by satan, ipsweep and so on
