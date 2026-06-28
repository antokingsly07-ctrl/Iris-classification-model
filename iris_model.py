
import csv
import math
import random
from collections import Counter


def euclidean_distance(point1, point2):
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

features = []
labels = []

with open('Iris.csv', mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        features.append([
            float(row['SepalLengthCm']),
            float(row['SepalWidthCm']),
            float(row['PetalLengthCm']),
            float(row['PetalWidthCm'])
        ])
        labels.append(row['Species'])

dataset = list(zip(features, labels))
random.seed(42) 
random.shuffle(dataset)

split_idx = int(len(dataset) * 0.8)
train_set = dataset[:split_idx]
test_set = dataset[split_idx:]

print(f"Training items: {len(train_set)}, Testing items: {len(test_set)}")

def predict_knn(train_data, test_instance, k=3):
    distances = []
    for train_features, train_label in train_data:
        dist = euclidean_distance(test_instance, train_features)
        distances.append((dist, train_label))
    
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    
    votes = [label for _, label in neighbors]
    most_common = Counter(votes).most_common(1)
    return most_common[0][0]

correct_predictions = 0

print("\n--- Making Predictions ---")
for test_features, actual_label in test_set:
    predicted_label = predict_knn(train_set, test_features, k=3)
    
    status = "✓" if predicted_label == actual_label else "✗"
    print(f"Predicted: {predicted_label.split('-')[-1]:<10} | Actual: {actual_label.split('-')[-1]:<10} {status}")
    
    if predicted_label == actual_label:
        correct_predictions += 1

accuracy = (correct_predictions / len(test_set)) * 100
print("\n" + "="*30)
print(f"Pure Python Model Accuracy: {accuracy:.2f}%")
print("="*30)

