import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

train = pd.read_csv("Train.csv")

X = train.drop(columns=["label", "attack_type", "num_outbound_cmds"])
y = train["label"]

#Splitting data and stratify is check equal mix of normal/attack vectors
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=47, stratify=y)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))

#Now we scale data so Logistic Regression multiplies each feature by a weight
#and adds them up and basically asks “how far above or below normal is this value?"
#and compares them across all columns as the pattern in data remains the same while units change
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

def show_results(model_name, y_true, y_pred):
    print("\n" + model_name)
    print("Accuracy :", round(accuracy_score(y_true, y_pred), 4))  #how many did it get right
    print("Precision:", round(precision_score(y_true, y_pred), 4)) #checks for false alarms
    print("Recall   :", round(recall_score(y_true, y_pred), 4))    #how many attacks get past
    print("F1-score :", round(f1_score(y_true, y_pred), 4))        #balancing precision and rcall

#MODEL 1 LOGISTICAL REGRESSION
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_val_scaled)
show_results("Logistic Regression", y_val, lr_pred)

#MODEL 2 RANDOM FOREST
rf = RandomForestClassifier(n_estimators=100, random_state=47)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_val)
show_results("Random Forest", y_val, rf_pred)

#MODEL 3 DECISION TREE
dt = DecisionTreeClassifier(max_depth=10, random_state=47)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_val)
show_results("Decision Tree", y_val, dt_pred)

#Saving output to be used in other parts during comparison
joblib.dump(lr, "logistic_regression.joblib")
joblib.dump(rf, "random_forest.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(dt, "decision_tree.joblib")