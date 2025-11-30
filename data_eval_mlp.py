
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from nn_train import FallDetectionMLP
import preprocessing as pp
import glob
import os
import sys

def prediction(filepath, model, max_len, scaler):
    # Preprocess (same as training)
    for file in filepath:
        raw_features = pp.dataprep(file)
        feature_array = np.array(raw_features, dtype=np.float64)
        relevant = np.concatenate([feature_array[:, 1:4], feature_array[:, 4:]], axis=1)
        
        # Scale using saved scaler
        scaled = scaler.transform(relevant)
        
        # Pad to same length as training
        padded = np.pad(scaled, ((0, max_len - len(scaled)), (0, 0)), 'constant')
        
        # Convert to tensor and predict
        tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            output = model(tensor)
            probability = torch.sigmoid(output)
            prediction = "FALL" if probability > 0.2 else "NOT FALL"
        
        print(file, prediction, probability)

if __name__ == '__main__':
    directory = sys.argv[1]

    parameters = torch.load('mlp_fall_detection.pth',weights_only=False)

    INPUT_SIZE = parameters['input_size']
    model = FallDetectionMLP(INPUT_SIZE, 256, 128, 1)
    model.load_state_dict(parameters['model_state_dict'])
    model.eval()

    scaler = parameters['scaler']
    max_len = parameters['max_len']

    TEST_DIR = directory
    test_files = glob.glob(os.path.join(TEST_DIR, '*.json'))
    
    prediction(test_files, model, max_len, scaler)