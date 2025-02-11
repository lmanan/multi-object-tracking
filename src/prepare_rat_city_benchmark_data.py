# based on ground true data and YOlO prediction data on same video,
# infer relations between detections for YOLO prediction by referring to ground true data.
# major idea: find closest detection matching the inference and ground true
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist



def map_yolo_to_gt_ids(tbyolo, tbgt, distance_threshold=50):
    """
    Map YOLO detection IDs to ground truth IDs based on spatial proximity and frame matching.
    
    Args:
        tbyolo: DataFrame with YOLO detections (id, frame_id, y, x, conf)
        tbgt: DataFrame with ground truth (id, frame_id, y, x, previous_id)
        distance_threshold: Maximum allowed distance for matching
        
    Returns:
        Dictionary mapping YOLO IDs to ground truth IDs
    """
    id_mapping = {}
    used_gt_ids = set()
    
    # Process each frame separately
    for frame in tbyolo['frame_id'].unique():
        # Get detections for current frame
        yolo_frame = tbyolo[tbyolo['frame_id'] == frame]
        gt_frame = tbgt[tbgt['frame_id'] == frame]
        
        if len(yolo_frame) == 0 or len(gt_frame) == 0:
            continue
            
        # Get coordinates
        yolo_points = yolo_frame[['x', 'y']].values  # x, y coordinates
        gt_points = gt_frame[['x', 'y']].values      # x, y coordinates
        
        # Calculate pairwise distances between all points
        distances = cdist(yolo_points, gt_points)
        
        # For each YOLO detection, find closest GT point
        for i, yolo_id in enumerate(yolo_frame['id'].values):
            min_dist_idx = np.argmin(distances[i])
            min_dist = distances[i][min_dist_idx]
            
            gt_id = gt_frame.iloc[min_dist_idx]['id']
            
            # Check if distance is within threshold and GT ID not already used
            if min_dist <= distance_threshold and gt_id not in used_gt_ids:
                id_mapping[yolo_id] = gt_id
                used_gt_ids.add(gt_id)
            else:
                id_mapping[yolo_id] = 0
    
    return id_mapping

# main
annotation_yolo_path = 'E:\\cohort2_exp1_yolo\\annotation_yolo\\annotation_yolo.txt'
annotation_GT_path = 'E:\\cohort2_exp1_yolo\\datasets\\annotation.txt'
savepath = 'E:\\cohort2_exp1_yolo\\annotation_yolo'

# Load ground truth annotations 
tbgt = pd.read_csv(annotation_GT_path, sep=' ')  # Now reading with headers
# Convert coordinate columns to numeric
tbgt['y'] = pd.to_numeric(tbgt['y'])
tbgt['x'] = pd.to_numeric(tbgt['x'])

# Load YOLO predictions 
tbyolo = pd.read_csv(annotation_yolo_path, sep=' ')  # Now reading with headers
# Convert coordinate columns to numeric
tbyolo['y'] = pd.to_numeric(tbyolo['y'])
tbyolo['x'] = pd.to_numeric(tbyolo['x'])

# Apply the mapping
id_mapping = map_yolo_to_gt_ids(tbyolo, tbgt)

# Create a new column in tbyolo with mapped ground truth IDs
tbyolo['gt_id'] = tbyolo['id'].map(id_mapping)

# Create a dictionary mapping gt_id to previous_id from tbgt
gt_to_prev_id = dict(zip(tbgt['id'], tbgt['previous_id']))

# Map previous_id using gt_id
tbyolo['previous_gt_id'] = tbyolo['gt_id'].map(gt_to_prev_id)
# Fill any missing values with 0
tbyolo['previous_gt_id'] = tbyolo['previous_gt_id'].fillna(0).astype(int)

# Create reverse mapping from GT IDs to YOLO IDs
reverse_id_mapping = {v: k for k, v in id_mapping.items() if v != 0}
# Create a new column in tbyolo with mapped ground truth IDs
tbyolo['previous_yolo_id'] = tbyolo['previous_gt_id'].map(reverse_id_mapping).fillna(0).astype(int)





# Save the results
output_path = f'{savepath}/annotation_yolo_with_gt_mapping.txt'
tbyolo.to_csv(output_path, sep=' ', index=False)
