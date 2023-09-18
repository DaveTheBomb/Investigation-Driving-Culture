# Execute the bash file to create the data file and download model
./script.sh
OR 
/script.sh
OR
script.sh

# Use the command below to execute the script

python script.py --source_weights_path data/traffic_analysis.pt --source_video_path data/traffic_analysis.mov --confidence_threshold 0.3 --iou_threshold 0.5 --target_video_path data/traffic_analysis_result.mov
