This entire folder can be directly uploaded to the Cloud Storage Bucket to start training
1. Label Map
2. Train TF Record
3. Test TF Record
4. Pre-trained model Checkpoint files ( http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz )
5. Training config file



# Create TF Record for training data set
python /path/to/create_table_tf_record.py \
    --label_map_path=/path/to/table_label_map.pbtxt \
    --data_dir=/path/to/data/folder/ --set=train \
    --output_path=table_train.record
	
# Create TF Record for Test data set	
python /path/to/create_table_tf_record.py \
    --label_map_path=/path/to/table_label_map.pbtxt \
    --data_dir=/path/to/data/folder/ --set=val \
    --output_path=table_val.record

# Submit Training job in google cloud shell 
gcloud ml-engine jobs submit training `whoami`_object_detection_`date +%s` \
    --job-dir=$BUCKET_NAME/train \
    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz \
    --module-name object_detection.train \
    --runtime-version 1.8 \
    --region us-central1 \
    --config object_detection/samples/cloud/cloud.yml \
    -- \
    --train_dir=$BUCKET_NAME/train \
    --pipeline_config_path=$BUCKET_NAME/data/ssd_mobilenet_v1_table.config
	
	
# Submit Test job in google cloud shell 	
gcloud ml-engine jobs submit training `whoami`_object_detection_eval_`date +%s` \
    --job-dir=$BUCKET_NAME/train \
    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,dist/pycocotools-2.0.tar.gz \
    --module-name object_detection.eval \
    --runtime-version 1.8 \
    --region us-east1 \
    --scale-tier BASIC_GPU \
    -- \
    --checkpoint_dir=$BUCKET_NAME/train \
    --eval_dir=$BUCKET_NAME/eval \
    --pipeline_config_path=$BUCKET_NAME/data/ssd_mobilenet_v1_table.config

# Export the trained model for inference
# Choose the latest checkpoint number and replace with ##### below	
python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path $BUCKET_NAME/train/pipeline.config \
    --trained_checkpoint_prefix $BUCKET_NAME/train/model.ckpt-##### \
    --output_directory /output/path/
