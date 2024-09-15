import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import imageio
import os
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

torch.cuda.empty_cache()

def get_gpu_info():
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        print("CUDA is not available.")

def get_gpu_memory_usage():
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Allocated Memory: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
            print(f"  Cached Memory: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
    else:
        print("CUDA is not available.")


get_gpu_info()
print('\n')
get_gpu_memory_usage()


# ---------- Load in the data ----------

# Load in the optical flow npz data

data_path = '/home/armlab/Documents/soft_manipulation/npz_files/single_layer_data'

# Load optical flow magnitudes for calibrated DT
print("Calibrated Camera:")
calibrated_flow_mags_0 = np.load(f'{data_path}/calibrated_flow_data_0_layers.npz')['flow_data'][:,0,:,:,:]
print(f"Flow data for 0 layers has shape {calibrated_flow_mags_0.shape}")

calibrated_flow_mags_1 = np.load(f'{data_path}/calibrated_flow_data_1_layers.npz')['flow_data'][:,0,:,:,:]
print(f"Flow data for 1 layers has shape {calibrated_flow_mags_1.shape}")

calibrated_flow_mags_2 = np.load(f'{data_path}/calibrated_flow_data_2_layers.npz')['flow_data'][:,0,:,:,:]
print(f"Flow data for 2 layers has shape {calibrated_flow_mags_2.shape}")

calibrated_flow_mags_3 = np.load(f'{data_path}/calibrated_flow_data_3_layers.npz')['flow_data'][:,0,:,:,:]
print(f"Flow data for 3 layers has shape {calibrated_flow_mags_3.shape}")

# Split the data into 70% training, 15% validation, and 15% testing
num_trials_0 = calibrated_flow_mags_0.shape[0]
num_trials_1 = calibrated_flow_mags_1.shape[0]
num_trials_2 = calibrated_flow_mags_2.shape[0]
num_trials_3 = calibrated_flow_mags_3.shape[0]

train_0 = calibrated_flow_mags_0[:int(0.70*num_trials_0)]
val_0 = calibrated_flow_mags_0[int(0.70*num_trials_0):int(0.85*num_trials_0)]
test_0 = calibrated_flow_mags_0[int(0.85*num_trials_0):]

train_1 = calibrated_flow_mags_1[:int(0.70*num_trials_1)]
val_1 = calibrated_flow_mags_1[int(0.70*num_trials_1):int(0.85*num_trials_1)]
test_1 = calibrated_flow_mags_1[int(0.85*num_trials_1):]

train_2 = calibrated_flow_mags_2[:int(0.70*num_trials_2)]
val_2 = calibrated_flow_mags_2[int(0.70*num_trials_2):int(0.85*num_trials_2)]
test_2 = calibrated_flow_mags_2[int(0.85*num_trials_2):]

train_3 = calibrated_flow_mags_3[:int(0.70*num_trials_3)]
val_3 = calibrated_flow_mags_3[int(0.70*num_trials_3):int(0.85*num_trials_3)]
test_3 = calibrated_flow_mags_3[int(0.85*num_trials_3):]

print("Calibrated Camera:")
print(f"Training data for 0 layers has shape {train_0.shape}")
print(f"Validation data for 0 layers has shape {val_0.shape}")
print(f"Testing data for 0 layers has shape {test_0.shape}")

print(f"Training data for 1 layers has shape {train_1.shape}")
print(f"Validation data for 1 layers has shape {val_1.shape}")
print(f"Testing data for 1 layers has shape {test_1.shape}")

print(f"Training data for 2 layers has shape {train_2.shape}")
print(f"Validation data for 2 layers has shape {val_2.shape}")
print(f"Testing data for 2 layers has shape {test_2.shape}")

print(f"Training data for 3 layers has shape {train_3.shape}")
print(f"Validation data for 3 layers has shape {val_3.shape}")
print(f"Testing data for 3 layers has shape {test_3.shape}")

# Concatenate and label train data
train_data = np.concatenate((train_0, train_1, train_2, train_3), axis=0)
train_data_tensor = torch.tensor(train_data).float().to(device)

labels_train = np.empty((train_data.shape[0]))
labels_train[:train_0.shape[0]] = 0
labels_train[train_0.shape[0]:train_0.shape[0]+train_1.shape[0]] = 1
labels_train[train_0.shape[0]+train_1.shape[0]:train_0.shape[0]+train_1.shape[0]+train_2.shape[0]] = 2
labels_train[train_0.shape[0]+train_1.shape[0]+train_2.shape[0]:] = 3

labels_train = torch.tensor(labels_train, dtype=torch.long).to(device)
print(f"Train Labels Shape: {labels_train.shape}")


# Concatenate and label validation data
val_data = np.concatenate((val_0, val_1, val_2, val_3), axis=0)
val_data_tensor = torch.tensor(val_data).float().to(device)

labels_val = np.empty((val_data.shape[0]))
labels_val[:val_0.shape[0]] = 0
labels_val[val_0.shape[0]:val_0.shape[0]+val_1.shape[0]] = 1
labels_val[val_0.shape[0]+val_1.shape[0]:val_0.shape[0]+val_1.shape[0]+val_2.shape[0]] = 2
labels_val[val_0.shape[0]+val_1.shape[0]+val_2.shape[0]:] = 3

labels_val = torch.tensor(labels_val, dtype=torch.long).to(device)
print(f"Validation Labels Shape: {labels_val.shape}")


# Concatenate and label test data
test_data = np.concatenate((test_0, test_1, test_2, test_3), axis=0)
test_data_tensor = torch.tensor(test_data).float().to(device)

labels_test = np.empty((test_data.shape[0]))
labels_test[:test_0.shape[0]] = 0
labels_test[test_0.shape[0]:test_0.shape[0]+test_1.shape[0]] = 1
labels_test[test_0.shape[0]+test_1.shape[0]:test_0.shape[0]+test_1.shape[0]+test_2.shape[0]] = 2
labels_test[test_0.shape[0]+test_1.shape[0]+test_2.shape[0]:] = 3

labels_test = torch.tensor(labels_test, dtype=torch.long).to(device)
print(f"Test Labels Shape: {labels_test.shape}")

get_gpu_memory_usage()


# ---------- Create the dataloaders ----------

from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(train_data_tensor, labels_train)
val_dataset = TensorDataset(val_data_tensor, labels_val)
test_dataset = TensorDataset(test_data_tensor, labels_test)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# ---------- Set up confusion matrices ----------

if not os.path.exists('train_confusion_matrices'):
    os.makedirs('train_confusion_matrices')

if not os.path.exists('test_confusion_matrices'):
    os.makedirs('test_confusion_matrices')

#  Create and save the confusion matrix
def plot_confusion_matrix(all_labels, all_preds, data_type, test_number=None): # data_type is either 'train' or 'test'
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3])
    disp.plot(cmap=plt.cm.Blues)
    if data_type == 'train':
        plt.title(f'Confusion Matrix at Epoch {epoch}')
        plt.savefig(f'{data_type}_confusion_matrices/{data_type}_confusion_matrix_epoch_{epoch}.png')
    else: # data_type == 'test'
        plt.title(f'Confusion Matrix at Test Number {test_number}')
        plt.savefig(f'{data_type}_confusion_matrices/{data_type}_confusion_matrix_test_{test_number}.png')
    plt.close()


# ----------- Create and train model -----------

class CNNFeatureExtractor(nn.Module):
    def __init__(self, output_size):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)  
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  
        self.fc1 = nn.Linear(128 * 6 * 8, 512)  
        self.fc2 = nn.Linear(512, output_size)  

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)  
        z = self.fc1(x) 
        x = self.fc2(z)
        return x, z

class CNNModel(nn.Module):
    def __init__(self, d_model, num_classes):
        super(CNNModel, self).__init__()
        self.feature_extractor = CNNFeatureExtractor(d_model)
        self.fc1 = nn.Linear(d_model, 40) 
        self.fc2 = nn.Linear(40, num_classes)

    def forward(self, cal_image_data):
        batch_size, seq_len, _, _, = cal_image_data.shape

        # calibrated optical flow
        cal_image_data = cal_image_data.view(batch_size * seq_len, 1, 96, 128)
        cal_image_features, _ = self.feature_extractor(cal_image_data)
        cal_image_features = cal_image_features.view(batch_size, seq_len, -1)

        x = cal_image_features.mean(dim=1)  # Pooling over the sequence dimension
        x = self.fc1(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1), x  # Return both the logits and the feature vector

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel(d_model=60, num_classes=4).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.00005)


# Training loop
max_epochs = 500
min_epochs = 300 # change to 500 to disable early stopping
loss_values = []
best_loss = float('inf')
early_stop_counter = 0
min_early_stop = 100
for epoch in range(max_epochs):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for cal_image_inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs, _ = model(cal_image_inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping
        optimizer.step()
        running_loss += loss.item()
        
        # Collect predictions and labels
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    # Condition to reset the optimizer state
    if epoch % 10 == 0:
        model_state = model.state_dict()
        optimizer = optim.Adam(model.parameters(), lr=0.00005)
        model.load_state_dict(model_state)
        print(f"Epoch {epoch}, Loss: {running_loss / len(train_loader)}")

    if epoch % 25 == 0:
        plot_confusion_matrix(all_labels, all_preds, 'train')
        
    epoch_loss = running_loss / len(train_loader)
    loss_values.append(epoch_loss)

    # Validation loop
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for cal_image_inputs, labels in val_loader:
            outputs, _ = model(cal_image_inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            running_val_loss += loss.item()
    avg_val_loss = running_val_loss / len(val_loader)

    # Save model with lowest validation loss
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        early_stop_counter = 0
        if not os.path.exists('best_model'):
            os.makedirs('best_model')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss / len(train_loader),
            }, 'best_model/best_model.pth')
        print(f"Current best model saved at epoch {epoch}, current best validation loss: {best_loss}")
    else:
        early_stop_counter += 1

    if early_stop_counter >= min_early_stop and epoch > min_epochs and np.abs(loss_values[-1] - loss_values[-2]) < 0.0001:
        print(f"Early stopping at epoch {epoch}")
        break

        
print('Finished Training')


# ---------- Create training confusion matrix video ----------

video_filename = 'train_confusion_matrices/train_confusion_matrix_vid.mp4'
png_files = sorted([f for f in os.listdir('train_confusion_matrices') if f.endswith('.png')])
png_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
with imageio.get_writer(video_filename, mode='I', fps=2) as writer:
    for png_file in png_files:
        image = imageio.imread(os.path.join('train_confusion_matrices', png_file))
        writer.append_data(image)


# ---------- Determine accuracy on test data ----------

def evaluate(model, test_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0
    test_number = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():  # Disable gradient computation
        for cal_image_inputs, labels in test_loader:
            cal_image_inputs, labels = cal_image_inputs.to(device), labels.to(device)
            outputs, _ = model(cal_image_inputs)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            formatted_output = "\n".join(["\t".join([f"{value:.4f}" for value in row]) for row in outputs])
            print(f"Test Number {test_number+1}: {formatted_output}")
            test_number += 1

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            plot_confusion_matrix(all_labels, all_predictions, 'test', test_number)

    avg_loss = test_loss / len(test_loader)
    accuracy = correct / total * 100

    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%')

    return avg_loss, accuracy, all_labels, all_predictions, test_number

# Load the best model for evaluation
checkpoint = torch.load('best_model/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
test_loss, test_accuracy, all_test_labels, all_test_predictions, final_test_number = evaluate(model, test_loader, criterion, device)


# ---------- Create test confusion matrix video ----------

test_video_filename = 'test_confusion_matrices/test_confusion_matrix_vid.mp4'
png_files = sorted([f for f in os.listdir('test_confusion_matrices') if f.endswith('.png')])
png_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
with imageio.get_writer(test_video_filename, mode='I', fps=2) as writer:
    for png_file in png_files:
        image = imageio.imread(os.path.join('test_confusion_matrices', png_file))
        writer.append_data(image)


# ---------- Plot the training loss over time ----------

def plot_losses(train_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train Loss over Epochs')

    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('plots/train_loss.png')

plot_losses(loss_values)


# ---------- Extract latent features and visualize with T-SNE ----------

def extract_latent_features(model, data_loader, device):
    model.eval()
    latent_features = []
    labels_list = []

    with torch.no_grad():
        for cal_image_inputs, labels in data_loader:
            cal_image_inputs, labels = cal_image_inputs.to(device), labels.to(device)
            _, latent = model(cal_image_inputs)        
            latent_features.append(latent.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    latent_features = np.concatenate(latent_features)
    labels_list = np.concatenate(labels_list)

    return latent_features, labels_list

latent_features, labels_list = extract_latent_features(model, test_loader, device)
np.save('plots/latent_features.npy', latent_features)
np.save('plots/labels_list.npy', labels_list)

# Apply T-SNE and plot
def plot_tsne(latent_features, labels_list, num_classes=4):
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(latent_features)

    plt.figure(figsize=(10, 7))
    for class_id in range(num_classes):
        indices = labels_list == class_id
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=f'Class {class_id}', alpha=0.5)

    if not os.path.exists('plots'):
        os.makedirs('plots')

    tsne_df = pd.DataFrame({
        'TSNE_1': tsne_results[:, 0],
        'TSNE_2': tsne_results[:, 1],
        'Label': labels_list
    })

    tsne_df.to_csv('plots/TSNE_values.csv', index=False)

    plt.legend()
    plt.title('T-SNE Plot of Latent Features')
    plt.xlabel('T-SNE 1')
    plt.ylabel('T-SNE 2')
    plt.savefig('plots/TSNE.png')

plot_tsne(latent_features, labels_list)


# ---------- Check the number of epochs the best model was saved at ----------

checkpoint = torch.load('best_model/best_model.pth')
print(f"Best model was saved at epoch {checkpoint['epoch']}")
