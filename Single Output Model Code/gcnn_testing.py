# main_script.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model_datagen import CustomDataset, GCNNModel
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

# Ensure the environment is properly configured for PyTorch
device = torch.device("cpu") #Change to "cuda" if you have a GPU and the correct cude version
print(f"Using device: {device}")

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
test_data_dir = os.path.join(script_dir, 'test_data')
num_files = len(os.listdir(test_data_dir))

# Parameters
num_classes = 1  # We are predicting a single parameter value. Can be changed to predict multiple values later on. For example if we add the J value
image_scaling = 90  # Reduces the amount of weights needing to be trained, I think this size is a sweet spot to not lose too many features
batch_size = num_files  # Use all files as one batch

# Load the saved model
model = GCNNModel(input_shape=(image_scaling, image_scaling), num_classes=num_classes).to(device)
model.load_state_dict(torch.load('GCNN_model_1.pth'))
model.eval()

# Test data loader
test_dataset = CustomDataset(
    directory=test_data_dir,
    target_size=(image_scaling, image_scaling),
    subset=None,
    validation_split=0.0  # No validation split for test data
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Evaluate the model
true_values = []
predictions = []
criterion = torch.nn.MSELoss()

with torch.no_grad():
    test_loss = 0.0
    for images, labels in test_loader:
        images = images.view(-1, 1, images.size(2), images.size(3)).to(device)
        labels = labels.view(-1).to(device).float()
        
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        
        predictions.extend(outputs.cpu().numpy())
        true_values.extend(labels.cpu().numpy())


r2 = r2_score(true_values, predictions)
print(f'R² Value: {r2}')


# Print absolute errors
ae = [np.abs(predictions[i] - true_values[i]) for i in range(len(predictions))]
for error in ae:
    print(error)

# Print true values and predictions
for true_value, prediction in zip(true_values, predictions):
    print(f"True Value: {true_value}, Prediction: {prediction}")

# Ensure true_values and predictions have the same size
if len(true_values) == len(predictions):
    plt.figure()
    plt.scatter(true_values, predictions, alpha=0.5)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'GCNN model, DMI parameter. R2 : {r2}')
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], color='red')  # Diagonal line for reference
    plt.savefig('tvp_GCNN2.png')  # Save the plot as a file
    # plt.show()
else:
    print("Mismatch in number of true values and predictions.")