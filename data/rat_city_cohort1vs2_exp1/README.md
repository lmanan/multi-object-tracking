### Summary
This folder has annotation files for the rat city experiments, subfolder YOLO11x and YOLO12x contains model predictions by YOLO11x and YOLO12x model respectively.
### Explanation of each file
original videos are saved a series of frames in `nrs:\karpova\forCedric\gnn\rat_city\cohort1vs2_exp1\datasets\images\train`  
`nrs:\karpova\forCedric\gnn\rat_city\cohort1vs2_exp1\datasets\annotation.mp4`  

`annotation-gt.txt` contains the ground truth annotation for each object's location  
header: id frame_id y x previous_id  
note:   
id unique id of detection, starting from 1
frame_id id of frame, starting from 1
y, x center of the object, normalized to 0-1 relative to the position in frame
previous_id id of same object in previous frame, 0 if no previous existence  

Folder YOLO11x and YOLO12x contains model prediction by YOLO11x and YOLO12x  
`annotation_yolo_with_gt_mapping.txt` contains yolo detection inference on same dataset  
`header`: id frame_id y x conf gt_id previous_gt_id previous_yolo_id  
note:
id unique id of detection, starting from 1
frame_id id of frame, starting from 1
y, x center of the object, normalized to 0-1 relative to the position in frame
conf confidence of the detection inference given by the yolo model
gt_id since this file is inference of location with same video of ground true, this indicates the id of corresponding object in ground true  
previous_gt_id previous id for the gt_id based on ground true (annotation.txt)
previous_yolo_id id of same object in previous frame, 0 if no previous existence. Inferred based on ground true.  

`annotation_yolo_with_gt_mapping_motile.csv`
header: id frame y x prev_id
note: output of script motile applied to annotation_yolo_with_gt_mapping.txt
python ../src/infer.py --yaml_configs_file_name rat_city_raw.yaml

`objects_per_frame_histogram.png` overview of number of detections across frames.
