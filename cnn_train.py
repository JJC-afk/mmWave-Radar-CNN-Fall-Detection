
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import glob
import random

import preprocessing as pp

#Convolutional Neural Network
import torch
import torch.nn as nn

class FallDetectionCNN(nn.Module):
    def __init__(self, num_features, hidden_size1, hidden_size2, output_size):
        """
        Args:
            num_features (int): The number of features per time step (e.g., 7).
                                This is the 'channels' for the Conv1d.
            hidden_size1 (int): Size of the first dense layer (e.g., 256).
            hidden_size2 (int): Size of the second dense layer (e.g., 128).
            output_size (int): Final output size (e.g., 1).
        """
        super(FallDetectionCNN, self).__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=21, padding=10),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
       
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(64 * 2, hidden_size1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        
        # --- 0. Permute ---
        # Conv1d expects [batch, channels, length]
        # So we must permute (transpose) our input.
        x = x.permute(0, 2, 1) 
        # New shape: [batch_size, num_features, max_len]
        
        # --- 1. Pass through Conv Blocks ---
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        
        # Get both max and average pooled features
        x_max = self.global_max_pool(x) # Shape: [batch, 64, 1]
        x_avg = self.global_avg_pool(x) # Shape: [batch, 64, 1]

        # Concatenate them along the channel dimension (dim=1)
        x = torch.cat((x_max, x_avg), dim=1) # Shape: [batch, 128, 1]

        x = self.flatten(x)     # Shape: [batch, 128]
        
        # --- 3. Pass through Classifier ---
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x

def load_data_from_folders(fall_dir, not_fall_dir):
    """Loads file paths and assigns labels (1 for fall, 0 for not fall)."""
    filepaths = []
    labels = []
    
    # Get all .json files from the fall directory
    fall_files = sorted(glob.glob(os.path.join(fall_dir, '*.json')))
    filepaths.extend(fall_files)
    labels.extend([1] * len(fall_files))
    
    # Get all .json files from the not_fall directory
    not_fall_files = sorted(glob.glob(os.path.join(not_fall_dir, '*.json')))
    filepaths.extend(not_fall_files)
    labels.extend([0] * len(not_fall_files))
    
    return filepaths, labels

def data_diagnostic(filepaths,labels,num_features,feature_names):
    print("\n--- Feature Comparison: Fall vs Not-Fall ---")
    fall_features = {i: [] for i in range(num_features)}
    notfall_features = {i: [] for i in range(num_features)}

    for f, label in zip(filepaths, labels):
        raw = pp.dataprep(f)
        if raw is None:
            continue
        for frame in raw:
            for i in range(num_features):
                if label == 1:
                    fall_features[i].append(frame[i])
                else:
                    notfall_features[i].append(frame[i])

    print(f"\n{'Feature':<12} {'Fall Mean':>12} {'NotFall Mean':>12} {'Difference':>12}")
    print("-" * 52)
    for i, name in enumerate(feature_names):
        fall_mean = np.mean(fall_features[i])
        notfall_mean = np.mean(notfall_features[i])
        diff = fall_mean - notfall_mean
        print(f"{name:<12} {fall_mean:>12.3f} {notfall_mean:>12.3f} {diff:>12.3f}")

def process_files(files, scaler=None, fit_scaler=False):
    sequences = []
    for f in files:
        raw_features = pp.dataprep(f)
        feature_array = np.array(raw_features, dtype=np.float64)
        # Select relevant features
        relevant = np.concatenate([feature_array[:, 1:4], feature_array[:, 4:]], axis=1)
        sequences.append(relevant)
    
    if fit_scaler and scaler==None:
        scaler = MinMaxScaler()
        concatenated = np.concatenate(sequences, axis=0)
        scaler.fit(concatenated)
    scaled_sequences = [scaler.transform(seq) for seq in sequences]
    
    if fit_scaler == True:
        return scaled_sequences, scaler
    else:
        return scaled_sequences
    
def pad_sequences(sequences, max_len):
    padded = np.array([np.pad(seq, ((0, max_len - len(seq)), (0, 0)), 'constant') for seq in sequences])
    return padded

def train_single_fold(X_train, y_train, X_val, y_val, config, verbose=False):

    noise = 0.1
    jitter = np.random.normal(loc=0.0, scale=noise, size=X_train.shape)
    X_train_augmented = X_train + jitter

    X_train_combined = np.concatenate((X_train, X_train_augmented), axis=0)
    y_train_combined = np.concatenate((y_train, y_train), axis=0)
    
    X_train_tensor = torch.tensor(X_train_combined, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_combined, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), 
                              batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), 
                            batch_size=config['batch_size'], shuffle=False)
    
    num_features = X_train.shape[2]
    model = FallDetectionCNN(num_features, config['hidden1'], config['hidden2'], 1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(config['epochs']):
        #train
        model.train()
        for sequences, labels in train_loader:
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        #validate
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            if verbose:
                print(f" Early stopping at epoch {epoch+1}")
            break
    
    #find best model and save it
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model.eval()
    with torch.no_grad():
        #training acc
        train_preds, train_labels = [], []
        for sequences, labels in train_loader:
            outputs = model(sequences)
            predicted = (torch.sigmoid(outputs) > config['threshold']).float()
            train_preds.extend(predicted.numpy())
            train_labels.extend(labels.numpy())
        train_acc = accuracy_score(train_labels, train_preds)
        
        #validation acc
        val_preds, val_labels = [], []
        for sequences, labels in val_loader:
            outputs = model(sequences)
            predicted = (torch.sigmoid(outputs) > config['threshold']).float()
            val_preds.extend(predicted.numpy())
            val_labels.extend(labels.numpy())
        val_acc = accuracy_score(val_labels, val_preds)
        val_cm = confusion_matrix(val_labels, val_preds)
    
    return {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'val_loss': best_val_loss,
        'confusion_matrix': val_cm,
        'model_state': best_model_state,
        'model': model
    }

if __name__ == '__main__':
            
    FALL_DATA_DIR = 'data/fall'
    NOT_FALL_DATA_DIR = 'data/not_fall'

    config = {
        'hidden1': 256,
        'hidden2': 128,
        'lr': 0.001,
        'weight_decay': 1e-4,
        'batch_size': 16,
        'epochs': 40,
        'patience': 10,
        'threshold': 0.3,
        'n_folds': 10,
        'test_size': 0.3,
        'random_state': 42  # Fixed seed for reproducible test split
    }

    # --- 2. Load File Paths and Labels ---
    filepaths, labels = load_data_from_folders(FALL_DATA_DIR, NOT_FALL_DATA_DIR)
    filepath = np.array(filepaths)
    labels = np.array(labels)

    num_features = 9
    feature_names = ['num_objs', 'avg_x', 'range_x', 'avg_vel', 'std_vel', 'max_vel', 'min_vel', 'max_snr','accel']
    
    data_diagnostic(filepaths,labels,num_features,feature_names)
    
    if len(filepaths) < 4:
        print("Error: Not enough data. Please add multiple json files to 'data/fall' and 'data/not_fall' folders.")
    else:
        # K-Folds Cross validation
        train_val_files, test_files, y_train_val, y_test = train_test_split(
            filepaths, labels, test_size=config['test_size'], random_state=config['random_state'], stratify=labels
        )

        print(f"\n--- Dataset Split ---")
        print(f"Train+Validate: {len(train_val_files)} samples")
        print(f"Test: {len(test_files)} samples")

        print(f"\n--- {config['n_folds']}-Fold Cross-Validation ---")

        skf = StratifiedKFold(n_splits=config['n_folds'], shuffle=True, random_state=config['random_state'])

        fold_results = []
        all_val_cms = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_files, y_train_val)):
            print(f"\nFold {fold + 1}/{config['n_folds']}:")

            # Get files for this fold
            fold_train_files = [ train_val_files[index] for index in train_idx ]
            fold_val_files = [ train_val_files[index] for index in val_idx ]
            fold_y_train = [ y_train_val[index] for index in train_idx ]
            fold_y_val = [ y_train_val[index] for index in val_idx ]
            
            print(f" Train: {len(fold_train_files)} ({sum(fold_y_train)} falls)")
            print(f" Val: {len(fold_val_files)} ({sum(fold_y_val)} falls)")
            
            # Process and scale data (fit scaler on training fold only)
            X_train_scaled, scaler = process_files(fold_train_files, fit_scaler=True)
            X_val_scaled = process_files(fold_val_files, scaler=scaler)

            max_len = max(len(seq) for seq in X_train_scaled + X_val_scaled)
            X_train_padded = pad_sequences(X_train_scaled, max_len)
            X_val_padded = pad_sequences(X_val_scaled, max_len)

            num_features = X_train_padded.shape[2]

            result = train_single_fold(
                X_train_padded, fold_y_train,
                X_val_padded, fold_y_val,
                config, verbose=True
            )

            fold_results.append(result)
            all_val_cms.append(result['confusion_matrix'])

            print(f" Train Acc: {result['train_acc']*100:.2f}%")
            print(f" Val Acc: {result['val_acc']*100:.2f}%")
            print(f" Confusion Matrix:\n{result['confusion_matrix']}")

        print("\nCROSS-VALIDATION SUMMARY")

        train_accs = [r['train_acc'] for r in fold_results]
        val_accs = [r['val_acc'] for r in fold_results]

        print(f"\nTraining Accuracy: {np.mean(train_accs)*100:.2f}% ± {np.std(train_accs)*100:.2f}%")
        print(f"Validation Accuracy: {np.mean(val_accs)*100:.2f}% ± {np.std(val_accs)*100:.2f}%")

        total_cm = sum(all_val_cms)
        print(f"\nAggregated Confusion Matrix (all validation folds):")
        print(total_cm)

        tn, fp, fn, tp = total_cm.ravel()
        total_samples = tn + fp + fn + tp
        print(f"True Negatives:  {tn} ({tn/total_samples*100:.1f}%)")
        print(f"False Positives: {fp} ({fp/total_samples*100:.1f}%) - false alarms")
        print(f"False Negatives: {fn} ({fn/total_samples*100:.1f}%) - MISSED FALLS")
        print(f"True Positives:  {tp} ({tp/total_samples*100:.1f}%)")
        print(f"Fall Recall:     {tp/(tp+fn)*100:.1f}% (sensitivity)")
        print(f"Fall Precision:  {tp/(tp+fp)*100:.1f}%" if (tp+fp) > 0 else "Fall Precision:  N/A")
    

        # --- Final Test Set Evaluation ---
        print("\nFINAL TEST SET EVALUATION")
    
        # Retrain on ALL train_val data, test on held-out test set
        print("\nRetraining on full train+val set...")

        X_trainval_scaled, final_scaler = process_files(train_val_files, fit_scaler=True)
        X_test_scaled = process_files(test_files, scaler=final_scaler)

        max_len = max(len(seq) for seq in X_trainval_scaled + X_test_scaled)
        X_trainval_padded = pad_sequences(X_trainval_scaled, max_len)
        X_test_padded = pad_sequences(X_test_scaled, max_len)
        
        num_features = X_trainval_padded.shape[2]
        
        # Use a fixed validation split from train_val for early stopping during final training
        final_train_idx = int(len(X_trainval_padded) * 5/6)
        X_final_train = X_trainval_padded[:final_train_idx]
        y_final_train = y_train_val[:final_train_idx]
        X_final_val = X_trainval_padded[final_train_idx:]
        y_final_val = y_train_val[final_train_idx:]
        
        final_result = train_single_fold(
            X_final_train, y_final_train,
            X_final_val, y_final_val,
            config, verbose=True
        )
        
        # Evaluate on test set
        final_model = final_result['model']
        final_model.eval()
        
        X_test_tensor = torch.tensor(X_test_padded, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.tolist(), dtype=torch.float32).unsqueeze(1)
        test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=config['batch_size'], shuffle=False)
        
        with torch.no_grad():
            test_preds, test_labels = [], []
            for sequences, labels_batch in test_loader:
                outputs = final_model(sequences)
                predicted = (torch.sigmoid(outputs) > config['threshold']).float()
                test_preds.extend(predicted.numpy())
                test_labels.extend(labels_batch.numpy())
            
            test_acc = accuracy_score(test_labels, test_preds)
            test_cm = confusion_matrix(test_labels, test_preds)
        
        print(f"\nTest Accuracy: {test_acc*100:.2f}%")
        print(f"Test Confusion Matrix:")
        print(test_cm)
        
        tn, fp, fn, tp = test_cm.ravel()
        print(f"\nTest Metrics:")
        print(f"True Negatives:  {tn}")
        print(f"False Positives: {fp} - false alarms")
        print(f"False Negatives: {fn} - MISSED FALLS")
        print(f"True Positives:  {tp}")
        print(f"Fall Recall:     {tp/(tp+fn)*100:.1f}%" if (tp+fn) > 0 else "Fall Recall: N/A")
        print(f"Fall Precision:  {tp/(tp+fp)*100:.1f}%" if (tp+fp) > 0 else "Fall Precision:  N/A")
        
        # --- Summary ---
        print("\nFINAL SUMMARY")
        print(f"Validation Accuracy: {np.mean(val_accs)*100:.2f}% ± {np.std(val_accs)*100:.2f}%")
        print(f"Test Accuracy:          {test_acc*100:.2f}%")
        print(f"Missed Falls (Test):      {fn} out of {tp+fn}")
        print(f"False Alarms (Test):      {fp} out of {tn+fp}")

        torch.save({
            'model_state_dict': final_model.state_dict(),
            'scaler': scaler,
            'num_features': num_features,
            'max_len': max_len,
            'seed': 42,
            'test_accuracy': test_acc,
            }, 
        'cnn_fall_detection.pth')