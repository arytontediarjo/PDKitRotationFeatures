# Gait Feature Extraction using PDKit & Rotation Features

<b> Author: Aryton Tediarjo, Larsson Omberg </b>

This repository is used for retrieving gait features from accelerometer and gyroscope sensor data based on PDKit walking features with some extra functionality and QC filtering (rotation and variance cutoff).  

## Added Functionality
- Additional PDKit Error Handling
- Added windowing data processing
- Added rotation detection

## Installation
```bash
pip install git+https://github.com/arytontediarjo/PDKitRotationFeatures.git
```

## Use Cases
```python
from PDKitRotationFeatures import gait_module  
featureObjs = gait_module.GaitFeaturize(window_size = 256) ##refer to module for additional parameter
featureObjs.run_gait_feature_pipeline(accel sensor data <t,x,y,z>, gyro sensor data <t,x,y,z>)
```

## Feature Dictionaries



## Citations


