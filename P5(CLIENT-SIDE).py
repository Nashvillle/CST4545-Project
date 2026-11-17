import socket
import json
import time
import pandas as pd

HOST = "127.0.0.1"
PORT = 5000

# Pick 10 random records from the test set to act as incoming traffic
test = pd.read_csv("Test.csv")
sample = test.sample(n=10, random_state=47)

actual_labels = sample["label"].tolist()
attack_types = sample["attack_type"].tolist()
features = sample.drop(columns=["label", "attack_type", "num_outbound_cmds"])
records = features.to_dict(orient="records")

# Connect to the IDS server
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((HOST, PORT))
print("Connected to the IDS server\n")

correct = 0
for i in range(len(records)):
    # Send one record and wait for the server's answer
    message = json.dumps(records[i])
    client.sendall(message.encode())
    result = client.recv(1024).decode()

    if actual_labels[i] == 1:
        actual = "ANOMALY"
    else:
        actual = "NORMAL"

    if result == actual:
        correct = correct + 1

    print(f"Record {i + 1}: actual = {actual} ({attack_types[i]}), server says = {result}")
    time.sleep(1)

print(f"\n{correct} out of {len(records)} records classified correctly")
client.close()