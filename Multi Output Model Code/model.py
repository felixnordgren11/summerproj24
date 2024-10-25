import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import re
import cv2
import matplotlib.pyplot as plt
import e2cnn.nn as gnn
from e2cnn.gspaces import Rot2dOnR2

# Set the data directory path
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'clean_training')

class MultiOutputGCNNModel(gnn.SequentialModule):
    def __init__(self, input_shape=(140, 140)):
        super(MultiOutputGCNNModel, self).__init__()

        # Define the rotation symmetry group p4
        r2_act = Rot2dOnR2(N=4)

        # Define the input type
        in_type = gnn.FieldType(r2_act, 1 * [r2_act.trivial_repr])  # 1 input channel (grayscale)

        # Start with 16 filters and double after each pooling
        filters = [16, 32, 64, 128, 256]

        # Define convolutional layers with max pooling after 2nd, 4th, 7th, and 10th layers
        self.input_type = in_type
        self.conv_layers = gnn.SequentialModule(
            gnn.R2Conv(in_type, gnn.FieldType(r2_act, filters[0] * [r2_act.regular_repr]), kernel_size=3, padding=1),
            gnn.ReLU(gnn.FieldType(r2_act, filters[0] * [r2_act.regular_repr])),

            gnn.R2Conv(gnn.FieldType(r2_act, filters[0] * [r2_act.regular_repr]), gnn.FieldType(r2_act, filters[0] * [r2_act.regular_repr]), kernel_size=3, padding=1),
            gnn.ReLU(gnn.FieldType(r2_act, filters[0] * [r2_act.regular_repr])),
            gnn.PointwiseMaxPool(gnn.FieldType(r2_act, filters[0] * [r2_act.regular_repr]), kernel_size=2, stride=2),  # Pool after 2nd layer

            gnn.R2Conv(gnn.FieldType(r2_act, filters[0] * [r2_act.regular_repr]), gnn.FieldType(r2_act, filters[1] * [r2_act.regular_repr]), kernel_size=3, padding=1),
            gnn.ReLU(gnn.FieldType(r2_act, filters[1] * [r2_act.regular_repr])),

            gnn.R2Conv(gnn.FieldType(r2_act, filters[1] * [r2_act.regular_repr]), gnn.FieldType(r2_act, filters[1] * [r2_act.regular_repr]), kernel_size=3, padding=1),
            gnn.ReLU(gnn.FieldType(r2_act, filters[1] * [r2_act.regular_repr])),
            gnn.PointwiseMaxPool(gnn.FieldType(r2_act, filters[1] * [r2_act.regular_repr]), kernel_size=2, stride=2),  # Pool after 4th layer

            gnn.R2Conv(gnn.FieldType(r2_act, filters[1] * [r2_act.regular_repr]), gnn.FieldType(r2_act, filters[2] * [r2_act.regular_repr]), kernel_size=3, padding=1),
            gnn.ReLU(gnn.FieldType(r2_act, filters[2] * [r2_act.regular_repr])),

            gnn.R2Conv(gnn.FieldType(r2_act, filters[2] * [r2_act.regular_repr]), gnn.FieldType(r2_act, filters[2] * [r2_act.regular_repr]), kernel_size=3, padding=1),
            gnn.ReLU(gnn.FieldType(r2_act, filters[2] * [r2_act.regular_repr])),

            gnn.R2Conv(gnn.FieldType(r2_act, filters[2] * [r2_act.regular_repr]), gnn.FieldType(r2_act, filters[2] * [r2_act.regular_repr]), kernel_size=3, padding=1),
            gnn.ReLU(gnn.FieldType(r2_act, filters[2] * [r2_act.regular_repr])),

            gnn.R2Conv(gnn.FieldType(r2_act, filters[2] * [r2_act.regular_repr]), gnn.FieldType(r2_act, filters[3] * [r2_act.regular_repr]), kernel_size=3, padding=1),
            gnn.ReLU(gnn.FieldType(r2_act, filters[3] * [r2_act.regular_repr])),
            gnn.PointwiseMaxPool(gnn.FieldType(r2_act, filters[3] * [r2_act.regular_repr]), kernel_size=2, stride=2),  # Pool after 7th layer

            gnn.R2Conv(gnn.FieldType(r2_act, filters[3] * [r2_act.regular_repr]), gnn.FieldType(r2_act, filters[3] * [r2_act.regular_repr]), kernel_size=3, padding=1),
            gnn.ReLU(gnn.FieldType(r2_act, filters[3] * [r2_act.regular_repr])),

            gnn.R2Conv(gnn.FieldType(r2_act, filters[3] * [r2_act.regular_repr]), gnn.FieldType(r2_act, filters[3] * [r2_act.regular_repr]), kernel_size=3, padding=1),
            gnn.ReLU(gnn.FieldType(r2_act, filters[3] * [r2_act.regular_repr])),

            gnn.R2Conv(gnn.FieldType(r2_act, filters[3] * [r2_act.regular_repr]), gnn.FieldType(r2_act, filters[4] * [r2_act.regular_repr]), kernel_size=3, padding=1),
            gnn.ReLU(gnn.FieldType(r2_act, filters[4] * [r2_act.regular_repr])),
            gnn.PointwiseMaxPool(gnn.FieldType(r2_act, filters[4] * [r2_act.regular_repr]), kernel_size=2, stride=2)  # Pool after 10th layer
        )

        # Calculate the flattened size accurately
        final_feature_map_size = (input_shape[0] // 16, input_shape[1] // 16)
        flattened_size = 4 * filters[4] * final_feature_map_size[0] * final_feature_map_size[1]

        # Define the fully connected layers
        self.dense_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),

            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),

            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
        )

        # Define the unified output branch
        self.output_branch = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # Output layer for 2 parameters (dm and j0)
        )
    
    
    def forward(self, x):
        x = gnn.GeometricTensor(x, self.input_type)  
        x = self.conv_layers(x)
        x = x.tensor 
        x = self.dense_layers(x)
        outputs = self.output_branch(x)
        return outputs[:, 0], outputs[:, 1]
    


        



# Function to preprocess (sharpen) the image
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened


# Helper function to extract parameters from filename
def extract_parameters_from_filename(filename):
    match = re.search(r'MLdata_dm_(-?\d{1}\.\d{2})_j0_(\d{1}\.\d{2})\.png', filename)
    if match:
        dm_value = float(match.group(1))
        j0_value = float(match.group(2))
        return dm_value, j0_value
    return None, None

# Custom Dataset
class CustomDataset(data.Dataset):
    def __init__(self, directory=data_dir, target_size=(128, 128), subset='training', validation_split=0.2):
        self.directory = directory
        self.target_size = target_size
        self.subset = subset
        self.filenames = [f for f in os.listdir(directory) if f.endswith('.png')]
        self.validation_split = validation_split
        self.n = len(self.filenames)
        self.indexes = np.arange(self.n)
        self.split_index = int(self.n * (1 - self.validation_split))
        if self.subset == 'training':
            self.indexes = self.indexes[:self.split_index]
        elif self.subset == 'validation':
            self.indexes = self.indexes[self.split_index:]
        np.random.shuffle(self.indexes) #Comment out when testing model
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.target_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        img_name = self.filenames[self.indexes[index]]
        img_path = os.path.join(self.directory, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = sharpen_image(image)
        image = self.transform(image)

        dm_value, j0_value = extract_parameters_from_filename(img_name)

        return image, torch.tensor(dm_value, dtype=torch.float32), torch.tensor(j0_value, dtype=torch.float32)

# Data Loaders
def get_dataloader(directory, batch_size=16, target_size=(128, 128), validation_split=0.2):
    train_dataset = CustomDataset(directory, target_size, subset='training', validation_split=validation_split)
    val_dataset = CustomDataset(directory, target_size, subset='validation', validation_split=validation_split)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader


