import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader, random_split
from gcnn_model import MultiOutputGCNNModel, CustomDataset
import time

start_time = time.time()


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
USE_CUDA = False
device = 'cpu'
if USE_CUDA:
    device = 'cuda'
print(f"Using device: {device}")

num_classes = 1  # We are predicting a single parameter value. Can be changed to predict multiple values later on. For example if we add the J value
image_scaling = 140  # 90 works  # Reduces the amount of weights needing to be trained, I think this size is a sweet spot to not lose too many features
batch_size = 16  # (16) gave an ok result, could try lowering to introduce more noise
epochs = 50
learning_rate = 0.00004 

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'For_Felix_4/_training') #Make sure this is correct

if not os.path.isdir(data_dir):
    raise FileNotFoundError(f"The directory {data_dir} does not exist.")

input_shape = (image_scaling, image_scaling)

gcnn_model = MultiOutputGCNNModel(input_shape=input_shape).to(device)

#split up dataset
dataset = CustomDataset(
    directory=data_dir,
    target_size=(image_scaling, image_scaling),
    subset='all',
    validation_split=0.2
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# Choose loss function
#criterion = nn.SmoothL1Loss()
criterion = nn.MSELoss()
optimizer = optim.Adam(gcnn_model.parameters(), lr=learning_rate)
train_losses = []
val_losses = []

# Train the model
for epoch in range(epochs):
    gcnn_model.train()
    running_loss = 0.0
    
    for images, dm_labels, j0_labels in train_loader:
        images = images.view(-1, 1, images.size(2), images.size(3)).to(device)
        dm_labels = dm_labels.view(-1).to(device).float()
        j0_labels = j0_labels.view(-1).to(device).float()

        optimizer.zero_grad()

        # Forward pass
        dm_outputs, j0_outputs = gcnn_model(images)
        
        # Combine the outputs and labels into a single tensor with correct dimensions
        combined_outputs = torch.cat((dm_outputs.unsqueeze(1), j0_outputs.unsqueeze(1)), dim=1)
        combined_labels = torch.cat((dm_labels.unsqueeze(1), j0_labels.unsqueeze(1)), dim=1)
        
        # Compute the loss
        loss = criterion(combined_outputs, combined_labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))

    # Validation
    gcnn_model.eval()
    val_loss = 0.0
    val_predictions = []
    val_true = []
    
    with torch.no_grad():
        for images, dm_labels, j0_labels in val_loader:
            images = images.view(-1, 1, images.size(2), images.size(3)).to(device)
            dm_labels = dm_labels.view(-1).to(device).float()
            j0_labels = j0_labels.view(-1).to(device).float()
            
            dm_outputs, j0_outputs = gcnn_model(images)
            
            combined_outputs = torch.cat((dm_outputs.unsqueeze(1), j0_outputs.unsqueeze(1)), dim=1)
            combined_labels = torch.cat((dm_labels.unsqueeze(1), j0_labels.unsqueeze(1)), dim=1)
            
            # Compute validation loss
            loss = criterion(combined_outputs, combined_labels)
            val_loss += loss.item()
            
            val_predictions.extend(combined_outputs.cpu().numpy())
            val_true.extend(combined_labels.cpu().numpy())

    val_losses.append(val_loss / len(val_loader))

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

val_mae = mean_absolute_error(val_true, val_predictions)
print(f"Validation MAE: {val_mae}")


torch.save(gcnn_model.state_dict(), 'GCNN_10conv_dataset_4.pth')
print("Model saved.")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Over Epochs')
plt.savefig('losses_exp.png')

end_time = time.time()
execution_time = end_time - start_time

print(f"Execution time: {execution_time:.1f} seconds")
