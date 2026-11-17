import socket
import json
import joblib
import pandas as pd

HOST = "127.0.0.1"
PORT = 5000

# Load the trained model from Part 2
model = joblib.load("random_forest.joblib")
print("Model loaded")

# Create the server socket and wait for a client
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen()
print("IDS server is waiting for a client on", HOST, PORT)

conn, addr = server.accept()
print("Client connected from", addr)

# Keep receiving records until the client disconnects
while True:
    data = conn.recv(4096)
    if not data:
        break

    record = json.loads(data.decode())
    record_df = pd.DataFrame([record])

    prediction = model.predict(record_df)[0]
    if prediction == 1:
        result = "ANOMALY"
    else:
        result = "NORMAL"

    print("Received a record -> classified as", result)
    conn.sendall(result.encode())

print("Client disconnected")
conn.close()
server.close()