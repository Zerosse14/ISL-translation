import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# Load data from the pickle file
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# Extract data and labels
data_list = data_dict['data']
labels = np.asarray(data_dict['labels'])

# Verify the unique labels to ensure we have all 49 classes
unique_labels = np.unique(labels)
print(f"Unique labels in the dataset: {unique_labels}")
assert len(unique_labels) == 49, f"There should be 49 unique classes, but there are {len(unique_labels)}."

# Flatten each element in data_list to a 1D array
flattened_data = [np.ravel(element) for element in data_list]

# Find the largest size in the flattened data
max_size = max(len(element) for element in flattened_data)

# Pad all data to match the largest size
padded_data = [
    np.pad(element, (0, max_size - len(element)), mode='constant')
    for element in flattened_data
]

# Convert the list to a NumPy array
data = np.asarray(padded_data)

# Ensure data and labels have consistent dimensions
if len(data) != len(labels):
    print(f"Mismatch detected! Data length: {len(data)}, Labels length: {len(labels)}")
    min_length = min(len(data), len(labels))
    data = data[:min_length]
    labels = labels[:min_length]
    print(f"Adjusted Data length: {len(data)}, Labels length: {len(labels)}")

# Check for class imbalance and compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weight_dict = {class_id: weight for class_id, weight in zip(np.unique(labels), class_weights)}
print(f"Class weights: {class_weight_dict}")

# Split data into training and testing sets (stratified sampling to keep the class distribution)
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Create and train the Random Forest Classifier with class weights
model = RandomForestClassifier(class_weight='balanced')  # Use class_weight='balanced' to handle imbalanced classes
model.fit(x_train, y_train)

# Make predictions on the test set
y_predict = model.predict(x_test)
print(f"Max size of padded data: {max_size}")

# Evaluate the model's accuracy
score = accuracy_score(y_predict, y_test)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the model to a pickle file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
