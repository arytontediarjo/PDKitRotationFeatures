# Gait Feature Extraction using PDKit & Rotation Features

**Author: Aryton Tediarjo, Larsson Omberg**

This repository is used for retrieving gait features from accelerometer and gyroscope sensor data based on PDKit walking features with some extra functionality and QC filtering (rotation and variance cutoff).  

## Added Functionality
- Additional PDKit Error Handling
- Added windowing data processing
- Added QC on detecting rotation 

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

## Feature Processing Information

In the feature processing steps, the module will try to detect rotation given a gyroscope information. After separating the rotation sequence from the longitudinal data, features will be computed in smaller windows (default to 512 ~ around 5 seconds given 100Hz sampling frequency).

These are the features that will be computed:

1. **Number of Steps**: The number of steps in a given window 

2. **Cadence**: The number of steps per window duration

3. **Energy Freeze Index**: Freeze Index is defined as the power in the “freeze” band [3–8 Hz] divided by the power in the “locomotor” band [0.5–3 Hz] [measured in Hz]

4. **Locomotor Freeze Index**: Locomotor freeze index is the power in the “freeze” band [3–8 Hz] added to power in the “locomotor” band [0.5–3 Hz]. [measured in Hz]

5. **Stride-to-Stride/Step-to-Step Features**: Compute step-to-step or stride-to-stride summary statistics (standard deviation, average etc.)


6. **Step/Stride Regularity**: Measure of step/stride regularity along axis [percentage consistency of the step-to-step pattern].

7. **Speed of Gait**: It extracts the gait speed from the energies of the approximation coefficients of wavelet functions. Preferably you should use the magnitude of x, y and z (mag_acc_sum) here, as the time series.

8. **Gait Symmetry**: Measure of gait symmetry along axis [difference between step and stride regularity].

9. **Rotation Speed (Omega)**: Measures rotation speed based on inferred rotation period (radians during inferred rotation period/rotation duration)

## Citations
1. Rotation-Detection Paper : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5811655/
2. PDKit Docs: https://pdkit.readthedocs.io/en/latest/gait.html
3. PDKit Gait Source Codes  : https://github.com/pdkit/pdkit
