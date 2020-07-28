# Gait Feature Extraction using PDKit & Rotation Features

<b> Author: Aryton Tediarjo, Larsson Omberg </b>

This repository is used for retrieving gait features from accelerometer and gyroscope sensor data based on PDKit walking features with some extra functionality and QC filtering (rotation and variance cutoff).  

## Added Functionality
- Additional PDKit Error Handling
- Added windowing data processing
- Added rotation detection

## Installation
```bash
pip install PDKitRotationFeatures
```

## Use Cases
```python
import PDKitRotationFeatures
gf = rotation_pdkit_gait_features.GaitFeaturize()
gf.run_gait_feature_pipeline(accel sensor data <t,x,y,z>, gyro sensor data <t,x,y,z>)
```

## Feature Dictionaries


## Citations

