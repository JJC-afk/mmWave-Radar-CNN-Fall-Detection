import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
import glob
import random

# Import the dataprepoc function from your preprocessing script
import preprocessing as pp

# --- 1. PyTorch Model Definition (FIXED) ---

import torch
import torch.nn as nn

class MyRNNCell(nn.Module):
   def __init__(self, input_size=1, hidden_size=64, num_layers=1):  
    super(MyRNNCell, self).__init__() 
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)  
    self.fc = nn.Linear(hidden_size, 1) 

   def forward(self, x): 
    out, _ = self.rnn(x)         # RNN output for all time steps 
    out = out[:, -1, :]          # Take output from the last time step 
    return self.fc(out)          # Pass through linear layer 


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

if __name__ == '__main__':
    # --- 1. Setup Data Directories ---
    if not os.path.exists('data/not_fall'):
        os.makedirs('data/not_fall')
        if os.path.exists('parsed.json'):
            os.rename('parsed.json', 'data/not_fall/sample1.json')

    if not os.path.exists('data/fall'):
        os.makedirs('data/fall')
        if not os.listdir('data/fall') and os.path.exists('data/not_fall/sample1.json'):
            import shutil
            shutil.copy('data/not_fall/sample1.json', 'data/fall/sample_fall_1.json')
            
    FALL_DATA_DIR = 'data/fall'
    NOT_FALL_DATA_DIR = 'data/not_fall'

    # --- 2. Load File Paths and Labels ---
    filepaths, labels = load_data_from_folders(FALL_DATA_DIR, NOT_FALL_DATA_DIR)
    
    if len(filepaths) < 4:
        print("Error: Not enough data. Please add multiple json files to 'data/fall' and 'data/not_fall' folders.")
    else:
        #testing split
        SEED = random.randint(42,999)
        train_val_files, test_files, y_train_val, y_test = train_test_split(
            filepaths, labels, test_size=0.1, random_state=SEED, stratify=labels
        )
        #validation & training split
        train_files, val_files, y_train, y_val = train_test_split(
            train_val_files, y_train_val, test_size=0.1, random_state=42, stratify=y_train_val
        )

        print(f"Dataset split: Train={len(train_files)}, Validation={len(val_files)}, Test={len(test_files)}")

        # --- 3. Process and Scale Training Data ---
        unscaled_train_sequences = []
        for f in train_files:
            raw_features = pp.dataprep(f)
            feature_array = np.array(raw_features, dtype=np.float64)
            # Select relevant features
            relevant = np.concatenate([feature_array[:, 1:4], feature_array[:, 4:]], axis=1)
            unscaled_train_sequences.append(relevant)

        scaler = MinMaxScaler()
        concatenated_train_data = np.concatenate(unscaled_train_sequences, axis=0)
        scaler.fit(concatenated_train_data)
        X_train_scaled = [scaler.transform(seq) for seq in unscaled_train_sequences]

        # --- 4. Process and Scale Testing Data ---
        X_test_scaled = []
        for f in test_files:
            raw_features = pp.dataprep(f)
            feature_array = np.array(raw_features, dtype=np.float64)
            relevant = np.concatenate([feature_array[:, 1:4], feature_array[:, 4:]], axis=1)
            X_test_scaled.append(scaler.transform(relevant))

        X_val_scaled = []
        for f in val_files:
            raw_features = pp.dataprep(f)
            feature_array = np.array(raw_features, dtype=np.float64)
            relevant = np.concatenate([feature_array[:, 1:4], feature_array[:, 4:]], axis=1)
            X_val_scaled.append(scaler.transform(relevant))

        # --- 5. Pad Sequences (Same as before) ---
        max_len = max(len(seq) for seq in X_train_scaled+X_test_scaled+X_val_scaled)
        
        X_train_padded = np.array([np.pad(seq, ((0, max_len - len(seq)), (0, 0)), 'constant') for seq in X_train_scaled])
        X_val_padded = np.array([np.pad(seq, ((0, max_len - len(seq)), (0, 0)), 'constant') for seq in X_val_scaled])
        X_test_padded = np.array([np.pad(seq, ((0, max_len - len(seq)), (0, 0)), 'constant') for seq in X_test_scaled])

        # --- 6. Prepare Data for PyTorch (Same as before) ---
        X_train_tensor = torch.tensor(X_train_padded, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

        X_val_tensor = torch.tensor(X_val_padded, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

        X_test_tensor = torch.tensor(X_test_padded, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        # ISSUE 4 FIXED: Smaller batch size for better gradient estimates with small datasets
        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=8, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=8, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=8, shuffle=False)

        # --- 7. Model, Loss, Optimizer, and Training ---
        INPUT_SIZE = X_train_padded.shape[2]
        hidden_size = 128
        num_layers = 2
        model = MyRNNCell(INPUT_SIZE, hidden_size, 1)
        
        # ISSUE 2 FIXED: Use BCEWithLogitsLoss for numerical stability
        num_fall = sum(y_train)
        num_not_fall = len(y_train) - num_fall
        pos_weight = torch.tensor([num_not_fall/num_fall])
        print(f"pos_weight: {pos_weight.item():.2f}")
        criterion = nn.BCEWithLogitsLoss()
        
        # ISSUE 5 FIXED: Adjusted learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)
        
        # ISSUE 6: Added learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        print("\n--- Starting PyTorch Model Training ---")
        print(model)
        print(f"Training samples: {len(X_train_tensor)}, Validation samples: {len(X_val_tensor)}, Test samples: {len(X_test_tensor)}")

        NUM_EPOCHS = 30

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        best_model_state = None
        
        # Track training history
        train_losses = []
        val_losses = []

        print("\n--- Feature Comparison: Fall vs Not-Fall ---")
        fall_features = {i: [] for i in range(7)}
        notfall_features = {i: [] for i in range(7)}

        feature_names = ['num_objs', 'avg_x', 'range_x', 'avg_vel', 'max_vel', 'min_vel', 'avg_snr']

        for f, label in zip(filepaths, labels):
            raw = pp.dataprep(f)
            if raw is None:
                continue
            for frame in raw:
                for i in range(7):
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
        
        for epoch in range(NUM_EPOCHS):
            # Training phase
            model.train()
            epoch_train_loss = 0
            for sequences, labels in train_loader:
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for sequences, labels in val_loader:
                    outputs = model(sequences)
                    loss = criterion(outputs, labels)
                    epoch_val_loss += loss.item()
            
            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # Update learning rate
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Restored best model (val_loss: {best_val_loss:.4f})")

        print("\n--- Training Complete ---")

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', color='blue')
        # plt.plot(val_losses, label='Validation Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        #plt.savefig('loss_plot.png')
        plt.show()

        # --- 8. Evaluation ---
        model.eval()
        with torch.no_grad():
            # Test set evaluation
            all_preds, all_labels = [], []
            for sequences, labels in test_loader:
                outputs = model(sequences)
                # Apply sigmoid for predictions since we removed it from the model
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                all_preds.extend(predicted.numpy())
                all_labels.extend(labels.numpy())
            
            test_accuracy = accuracy_score(all_labels, all_preds)

            # all_preds, all_labels = [], []
            # for sequences, labels in val_loader:
            #     outputs = model(sequences)
            #     # Apply sigmoid for predictions since we removed it from the model
            #     predicted = (torch.sigmoid(outputs) > 0.5).float()
            #     all_preds.extend(predicted.numpy())
            #     all_labels.extend(labels.numpy())
            
            # val_accuracy = accuracy_score(all_labels, all_preds)

            # Training set evaluation
            all_preds, all_labels = [], []
            for sequences, labels in train_loader:
                outputs = model(sequences)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                all_preds.extend(predicted.numpy())
                all_labels.extend(labels.numpy())
            
            train_accuracy = accuracy_score(all_labels, all_preds)

            print(f'\nAccuracy on Train Data: {train_accuracy * 100:.2f}%')
            # print(f'\nAccuracy on Validation Data: {val_accuracy * 100:.2f}%')
            print(f'\nAccuracy on Test Data: {test_accuracy * 100:.2f}%')
            
            # Check for overfitting
            gap = train_accuracy - test_accuracy
            if gap > 0.15:
                print(f"\nWarning: Model may be overfitting (gap: {gap*100:.1f}%)")
            elif gap < 0.05:
                print(f"\nModel generalizes well (gap: {gap*100:.1f}%)")

            # Add this after evaluation
            print(f"Predictions breakdown: {sum(all_preds)} fall, {len(all_preds) - sum(all_preds)} not-fall")
            print(f"Actual breakdown: {sum(all_labels)} fall, {len(all_labels) - sum(all_labels)} not-fall")