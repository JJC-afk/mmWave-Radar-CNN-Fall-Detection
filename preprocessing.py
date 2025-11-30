import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def dataprep(filepath='parsed.json'):
    """
    Loads radar data from the specified JSON file, extracts features from each
    frame, and scales them using MinMaxScaler.

    Args:
        filepath (str): The path to the JSON data file.

    Returns:
        np.ndarray: A 2D numpy array where each row is a scaled feature
                    vector for a single frame. Returns None if file not found.
    """
    try:
        with open(filepath, 'r') as f:
            # The actual data is nested under the "messages" key
            # and the first item is a raw string we need to skip.
            all_messages = json.load(f)['messages'][1:]
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error processing JSON file: {e}")
        return None

    all_frame_features = []
    frame_rate = 20

    # --- 1. Feature Extraction ---
    # Iterate through each frame message in the file
    for frame_data in all_messages:
        header = frame_data.get('header', {})
        body = frame_data.get('body', [])
        
        # Initialize features with default values for this frame
        num_objs = 0
        avg_x, range_x, avg_vel, max_vel, min_vel, max_snr = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        # Default range profile is 256 zeros
        # range_profile = [0] * 256 

        # Get the number of detected objects for this frame
        num_objs = header.get('numDetectedObj', 0)
        
        # If objects were detected, find their corresponding TLVs and aggregate data
        if num_objs > 0:
            # Find the TLV for detected objects (type 1)
            obj_tlv = next((tlv for tlv in body if tlv.get('header', {}).get('type') == 1), None)
            # Find the TLV for side information (snr, noise) (type 7)
            info_tlv = next((tlv for tlv in body if tlv.get('header', {}).get('type') == 7), None)

            if obj_tlv and info_tlv:
                obj_data = obj_tlv.get('body', {}).get('data', [])
                info_data = info_tlv.get('body', {}).get('data', [])

                # Ensure lists are not empty before calculating mean
                if obj_data:
                    moving_obj_data = [obj for obj in obj_data if (abs(obj.get('velocity',0)) > 0.1) and (abs(obj.get('x',0)) < 1.5)]
                    if moving_obj_data:
                        xvals = [obj.get('x', 0) for obj in moving_obj_data]
                        velocities = [abs(obj.get('velocity', 0)) for obj in moving_obj_data]

                        avg_x = np.mean(xvals)
                        range_x = max(xvals) - min(xvals)

                        avg_vel = np.mean(velocities)
                        std_vel = np.var(velocities)
                        max_vel = max(velocities)
                        min_vel = min(velocities)

                    else:
                        avg_x = np.mean([obj.get('x', 0) for obj in obj_data])
                        range_x = 0.0

                        avg_vel = 0.0
                        std_vel = 0.0
                        max_vel = 0.0
                        min_vel = 0.0

                if info_data:
                    max_snr = np.max([info.get('snr', 0) for info in info_data])
        else:
            continue
        
        # Extract the range profile data from TLV type 2 (should always exist)
        # range_tlv = next((tlv for tlv in body if tlv.get('header', {}).get('type') == 2), None)
        # if range_tlv:
        #     range_data = range_tlv.get('body', {}).get('data', [])
        #     # Pad with zeros if for some reason the profile is not 256 bins
        #     range_profile_values = [item.get('bin', 0) for item in range_data]
        #     range_profile = range_profile_values + [0] * (256 - len(range_profile_values))

        # Assemble the final feature vector for this frame (260 features)
        feature_vector = [num_objs, avg_x, range_x, avg_vel, std_vel, max_vel, min_vel, max_snr] #+ range_profile

        all_frame_features.append(feature_vector)

    for i in range(0, len(all_frame_features)):
        if (i == 0):
            if np.isnan(all_frame_features[i][1]):
                all_frame_features[i][1] = (all_frame_features[i+1][1])/2
            if np.isnan(all_frame_features[i][3]):
                all_frame_features[i][3] = (all_frame_features[i+1][3])/2
            if np.isnan(all_frame_features[i][6]):
                all_frame_features[i][6] = (all_frame_features[i+1][6])/2
        elif (i == len(all_frame_features) - 1):
            if np.isnan(all_frame_features[i][1]):
                all_frame_features[i][1] = (all_frame_features[i-1][1])/2
            if np.isnan(all_frame_features[i][3]):
                all_frame_features[i][3] = (all_frame_features[i-1][3])/2
            if np.isnan(all_frame_features[i][6]):
                all_frame_features[i][6] = (all_frame_features[i-1][6])/2
        else:
            if np.isnan(all_frame_features[i][1]):
                all_frame_features[i][1] = (all_frame_features[i-1][1] + all_frame_features[i+1][1])/2
            if np.isnan(all_frame_features[i][3]):
                all_frame_features[i][3] = (all_frame_features[i-1][3] + all_frame_features[i+1][3])/2
            if np.isnan(all_frame_features[i][6]):
                all_frame_features[i][6] = (all_frame_features[i-1][6] + all_frame_features[i+1][6])/2
    
    #acceleration
    vert_vel = []
    for frame in range( len(all_frame_features) ):
        if (frame == len(all_frame_features) - 1):
            vert_vel += [ (all_frame_features[frame][1] - all_frame_features[frame-1][1])*20 ]
        else:
            vert_vel += [ (all_frame_features[frame+1][1] - all_frame_features[frame][1])*20 ]
    
    window_size = 10  # Number of frames in window (5 frames = 0.25 sec at 20fps)
    accel = []

    for frame in range(len(all_frame_features)):
        # Get window boundaries
        start = max(0, frame - window_size // 2)
        end = min(len(vert_vel) - 1, frame + window_size // 2)
        
        # Compute acceleration as velocity change over window
        if end > start:
            accel_val = (vert_vel[end] - vert_vel[start]) * 20 / (end - start)
        else:
            accel_val = 0.0

        accel.append(accel_val)
        all_frame_features[frame].append(accel_val)

    # for frame in range( len(all_frame_features) ):
    #     all_frame_features[frame].append( max(accel) )
    #     all_frame_features[frame].append( min(accel) )
    #     all_frame_features[frame].append( max(accel) - min(accel) )

    return all_frame_features

