### Summary
This folder has annotation files for the rat city experiments
### Explanation of each file
original videos are saved a series of frames in nrs:\karpova\forCedric\gnn\rat_city\cohort2_exp1\datasets\images\train  

annotation.txt contains ground true annotation for each object's location  
header: id frame_id y x previous_id  
note:   
id unique id of detection, starting from 1
frame_id id of frame, starting from 1
y, x center of the object, normalized to 0-1 relative to the position in frame
previous_id id of same object in previous frame, 0 if no previous existence  

annotation_yolo_with_gt_mapping.txt contains yolo detection inference on same dataset  
header: id frame_id y x conf gt_id previous_gt_id previous_yolo_id
note:  
id unique id of detection, starting from 1
frame_id id of frame, starting from 1
y, x center of the object, normalized to 0-1 relative to the position in frame
conf confidence of the detection inference given by the yolo model
gt_id since this file is inference of location with same video of ground true, this indicates the id of corresponding object in ground true  
previous_gt_id previous id for the gt_id based on ground true (annotation.txt)
previous_yolo_id id of same object in previous frame, 0 if no previous existence. Inferred based on ground true.  



annotation_yolo_with_gt_mapping_motile.csv
header: id frame y x prev_id
note: output of script motile applied to annotation_yolo_with_gt_mapping.txt
python ../src/infer.py --yaml_configs_file_name rat_city_raw.yaml

