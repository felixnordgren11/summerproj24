
import os
import torch
import numpy as np
import re
import matplotlib.pyplot as plt
from gcnn_model import CustomDataset, MultiOutputGCNNModel
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

# Configure environment
device = torch.device("cpu")  # Change to "cuda" if using GPU
print(f"Using device: {device}")

## Helper function to extract parameters from filename
#def extract_parameters_from_filename(filename):
#    match = re.search(r'mompix\.dm_(\d{5})_j0_(\d{3})\.Z001\.png', filename)
#    if match:
#        dm_value = float(match.group(1))  # Extracting as int for exact matching
#        j0_value = float(match.group(2)) / 100.0
#        return dm_value, j0_value
#    return None, None


def extract_parameters_from_filename(filename):
    match = re.search(r'MLdata_dm_(-?\d{1}\.\d{2})_j0_(\d{1}\.\d{2})\.png', filename)
    if match:
        dm_value = float(match.group(1))
        j0_value = float(match.group(2))
        return dm_value, j0_value
    return None, None


script_dir = os.path.dirname(os.path.abspath(__file__))
test_data_dir = os.path.join(script_dir, "For_Felix_3/_testing") #Make sure to change dep. on dataset
num_files = len(os.listdir(test_data_dir))


image_scaling = 140  # Reduces the amount of weights needing to be trained, I think this size is a sweet spot to not lose too many features
batch_size = num_files  # Use all files as one batch


model = MultiOutputGCNNModel(input_shape=(image_scaling, image_scaling)).to(device)
model.load_state_dict(torch.load('GCNN_10conv_dataset_4.pth'))
model.eval()


test_dataset = CustomDataset(
    directory=test_data_dir,
    target_size=(image_scaling, image_scaling),
    subset=None,
    validation_split=0.0  
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

true_values = []
predictions = []
dm_values = []

criterion = torch.nn.MSELoss()

with torch.no_grad():
    test_loss = 0.0
    for images, dm_labels, j0_labels in test_loader:
        images = images.view(-1, 1, images.size(2), images.size(3)).to(device)
        dm_labels = dm_labels.view(-1).to(device).float()
        j0_labels = j0_labels.view(-1).to(device).float()
        
        dm_outputs, j0_outputs = model(images)
        
        # Combine the outputs and labels into a single tensor
        combined_outputs = torch.cat((dm_outputs.unsqueeze(1), j0_outputs.unsqueeze(1)), dim=1)
        combined_labels = torch.cat((dm_labels.unsqueeze(1), j0_labels.unsqueeze(1)), dim=1)
        
        # Compute the combined loss
        loss = criterion(combined_outputs, combined_labels)
        test_loss += loss.item()
        
        predictions.extend(combined_outputs.cpu().numpy())
        true_values.extend(combined_labels.cpu().numpy())
        



true_values = np.array(true_values)
predictions = np.array(predictions)

true_values_dm = true_values[:, 0]
true_values_j0 = true_values[:, 1]
predictions_dm = predictions[:, 0]
predictions_j0 = predictions[:, 1]

for i in range(len(true_values_j0)):
    print(f"dm: {true_values_dm[i]}, j0: (true {true_values_j0[i]}, pred {predictions_j0[i]})")


r2_dm = r2_score(true_values_dm, predictions_dm)
r2_j0 = r2_score(true_values_j0, predictions_j0)
print(f'R² Value (dm): {r2_dm}')
print(f'R² Value (j0): {r2_j0}')


ae_dm = np.abs(predictions_dm - true_values_dm)
ae_j0 = np.abs(predictions_j0 - true_values_j0)

for i in range(len(ae_dm)):
    print(f'Absolute Error (dm) for sample {i}: {ae_dm[i]}')

for i in range(len(ae_j0)):
    print(f'Absolute Error (j0) for sample {i}: {ae_j0[i]}')


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)

plt.scatter(np.array(true_values_dm), np.array(predictions_dm))
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title(f'dm, R2: {r2_dm}')
plt.plot([min(true_values_dm), max(true_values_dm)], [min(true_values_dm), max(true_values_dm)], color='red')  

plt.subplot(1, 2, 2)
plt.scatter(np.array(true_values_j0), np.array(predictions_j0))
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title(f'j0, R2: {r2_j0}')
plt.plot([min(true_values_j0), max(true_values_j0)], [min(true_values_j0), max(true_values_j0)], color='red')  
plt.suptitle('Dataset #3, trained on #4')

plt.savefig('tvp_gcnn_10conv_dataset_3_t4.png')
plt.show()