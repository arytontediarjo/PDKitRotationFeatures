# Gait Feature Extraction using PDKit & Rotation Features

Author: Aryton Tediarjo, Larsson Omberg

This repository is used for retrieving gait features from accelerometer and gyroscope based on external libraries and research implementations. 

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
gf.run_gait_feature_pipeline(accel sensor data <t,x,y,z>, gyro <t,x,y,z>)
```

## Feature Dictionaries


##

