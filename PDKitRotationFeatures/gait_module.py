"""
Author: Sage Bionetworks

CP: aryton.tediarjo@sagebase.org

About:
This module is primarily used as a wrapper for PDKit package and rotation detection on gait features. 
Additional featurization option added into this module is as follows:
    - Simple Sensor Filters (variance, resampling)
    - Rotation Detection on longitudinal data
    - Window feature computation for PDKit (number of steps per given window)

References:
    Rotation-Detection Paper : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5811655/
    PDKit Docs               : https://pdkit.readthedocs.io/_/downloads/en/latest/pdf/
    PDKit Gait Source Codes  : https://github.com/pdkit/gait_processor.py
    Freeze of Gait Docs      : https://ieeexplore.ieee.org/document/5325884
"""

# future liibrary imports
from __future__ import unicode_literals
from __future__ import print_function

# standard library import
import pandas as pd
import numpy as np
from sklearn import metrics

# pdkit imports
import pdkit
import pdkit.processor
from pdkit.utils import (butter_lowpass_filter,
                         numerical_integration)


class GaitFeatures:

    """
    Parameters used in gait features pipeline
    
    Parameters
    ----------
    sensor_variance_cutoff: minimal variance on longitudinal data
    sensor_window_size: desired window size
    sensor_window_overlap: percentage of overlap in window iteration (%)
    sensor_sampling_frequency: sampling frequency of data (Hz)
    detect_rotation: boolean input whether to detect rotation or not
    rotation_detection_axis: axis to detect rotation
    rotation_detection_frequency_cutoff: low-pass filter (Hz) followed based on paper
    rotation_detection_filter_order: filter order on user acceleration
    rotation_detection_aucXt_upper_limit: limit of AUC * time (metrics used in paper) for inferring rotation
    pdkit_gait_features_cutoff_frequency: low-pass filter on user acceleration for pdkit features
    pdkit_gait_features_filter_order: filter order on user acceleration for pdkit features
    pdkit_gait_features_delta: A point is considered a maximum peak if it has the maximal value, 
                                and was preceded (to the left) by a value lower by delta (0.5 default).
    pdkit_gait_features_low_freq: lower frequency limit in Hz (2.0 default)
    pdkit_gait_features_upper_freq: upper frequency limit in Hz (2.0 default)
    pdkit_gait_features_step_size: The average step size in centimeters (50.0 default).
    pdkit_gait_features_stride_fraction: <undocumented>
    fog_loco_band: The ratio of the energy in the locomotion band, measured in Hz ([0.5, 3] default)
    fog_freeze_band: The ration of energy in the freeze band, measured in Hz ([3, 8] default)

    Returns
    ----------
    Instantiated class object for gait features
    """


    def __init__(self,
                 sensor_variance_cutoff=1e-4,
                 sensor_window_size=512,
                 sensor_window_overlap=0.2,
                 sensor_sampling_frequency=100,
                 detect_rotation = True,
                 rotation_detection_axis = "y",
                 rotation_detection_frequency_cutoff=2,
                 rotation_detection_filter_order=2,
                 rotation_detection_aucXt_upper_limit=2,
                 pdkit_gait_features_cutoff_frequency=5,
                 pdkit_gait_features_filter_order=4,
                 pdkit_gait_features_delta=0.5,
                 pdkit_gait_features_low_freq = 2,
                 pdkit_gait_features_upper_freq = 10,
                 pdkit_gait_features_step_size = 50,
                 pdkit_gait_features_stride_fraction=0.125,
                 fog_loco_band=[0.5, 3],
                 fog_freeze_band=[3, 8]):
        
        ### parameters for sensor qc filtering
        self.sensor_sampling_frequency = sensor_sampling_frequency
        self.sensor_variance_cutoff = sensor_variance_cutoff
        self.sensor_window_size = sensor_window_size
        self.sensor_window_overlap = sensor_window_overlap
        self.sensor_sampling_frequency = sensor_sampling_frequency
        
        ### parameters for rotation detection
        self.detect_rotation = detect_rotation
        self.rotation_detection_axis = rotation_detection_axis
        self.rotation_detection_frequency_cutoff = rotation_detection_frequency_cutoff
        self.rotation_detection_filter_order = rotation_detection_filter_order
        self.rotation_detection_aucXt_upper_limit = rotation_detection_aucXt_upper_limit
        
        ### parameters for pdkit gait based on gait processor
        self.pdkit_gait_features_cutoff_frequency = pdkit_gait_features_cutoff_frequency
        self.pdkit_gait_features_filter_order = pdkit_gait_features_filter_order
        self.pdkit_gait_features_delta = pdkit_gait_features_delta
        self.pdkit_gait_features_low_freq = pdkit_gait_features_low_freq
        self.pdkit_gait_features_upper_freq = pdkit_gait_features_upper_freq
        self.pdkit_gait_features_step_size = pdkit_gait_features_step_size
        self.pdkit_gait_features_stride_fraction = pdkit_gait_features_stride_fraction
                
        ### parameters for freeze of gait
        self.fog_loco_band = fog_loco_band
        self.fog_freeze_band = fog_freeze_band
        
    def format_time_series(self, data):
        """
        Function to format mPower Dataframe after subselecting
        which sensor types it tries to query
        
        Parameters
        ----------
        data (dtype: pd.dataframe): sensor time-series with at least t,x,y,z info
        
        Returns
        ----------
        Returns a cleaned and resampled time-series
        """
        try:
            if data.empty:
                return "ERROR: Filepath has Empty Dataframe"
            elif not (set(data.columns) >= set(["x", "y", "z", "t"])):
                return "ERROR: [t, x, y, z] in columns is required"
            else:
                data = data.dropna(subset=["x", "y", "z"])
                data["AA"] = np.sqrt(data["x"]**2 + data["y"]**2 + data["z"]**2)
                data["td"] = data["t"] - data["t"].iloc[0]
                data["t"] = pd.to_datetime(data["td"], unit="s")
                data = data.set_index("t")
                data = data.sort_index()
                data = data[~data.index.duplicated(keep='first')]
                data = self.resample_signal(data[["td", "x", "y", "z", "AA"]])
            return(data)
        except(TypeError, AttributeError) as err:
            return "ERROR: %s" % type(err).__name__


    def resample_signal(self, dataframe):
        """
        Utility method for data resampling,
        Data will be interpolated using linear method

        Parameters
        ----------
        dataframe(dtype: pd.DataFrame): A time-indexed sensor dataframe

        Returns
        ----------
        Returns a pd.Dataframe based on defined sampling frequency
        """
        new_freq = np.round(1 / self.sensor_sampling_frequency, decimals=6)
        df_resampled = dataframe.resample(str(new_freq) + 'S').mean()
        df_resampled = df_resampled.interpolate(method='linear')
        return df_resampled


    def get_freeze_index_features(self, accel_series, sample_rate):
        """
        Modified pdkit FoG freeze index function to be compatible with
        current data gait pipeline
        
        Parameters
        ----------
        accel_series (dtype: pd.Series): pd.Series of acceleration signal in one axis
        sample_rate  (dtype: float)  : signal sampling rate
        
        Returns
        ----------
        Returns a list containing (energy freeze index, locomotor freeze index)
        """
        try:
            loco_band = self.fog_loco_band
            freeze_band = self.fog_freeze_band
            window_size = accel_series.shape[0]
            f_res = sample_rate / window_size
            f_nr_LBs = int(loco_band[0] / f_res)
            f_nr_LBe = int(loco_band[1] / f_res)
            f_nr_FBs = int(freeze_band[0] / f_res)
            f_nr_FBe = int(freeze_band[1] / f_res)

            # discrete fast fourier transform
            Y = np.fft.fft(accel_series, int(window_size))
            Pyy = abs(Y*Y) / window_size
            areaLocoBand = numerical_integration(
                Pyy[f_nr_LBs-1: f_nr_LBe],   sample_rate)
            areaFreezeBand = numerical_integration(
                Pyy[f_nr_FBs-1: f_nr_FBe], sample_rate)
            sumLocoFreeze = areaFreezeBand + areaLocoBand
            freezeIndex = areaFreezeBand / areaLocoBand

        except ZeroDivisionError:
            return "Not Enough Samples", "Not Enough Samples"

        else:
            if (freezeIndex == np.inf) or (freezeIndex == -np.inf):
                freezeIndex = np.NaN
            if (sumLocoFreeze == np.inf) or (sumLocoFreeze == -np.inf):
                sumLocoFreeze = np.NaN
            return freezeIndex, sumLocoFreeze
    
    def get_pdkit_gait_features(self, accel_dataframe):
        """
        Function to featurize dataframe using pdkit package (gait)
        based on all axis orientation (x, y, z, AA). 

        Note: Extra error handling in computing step/stride manually
              as pdkit will throw an error when stride is undetected.

        Features from pdkit contains the following:
            1. Number of steps
            2. Cadence
            3. Freeze Index
            4. Energy/Locomotor Freeze Index
            5. Average Step/Stride Duration
            6. Std of Step/Stride Duration
            7. Step/Stride Regularity
            8. Speed of Gait
            9. Symmetry

        Parameters
        -----------
        accel_dataframe (type: pd.DataFrame): user acceleration information containing axis (t (index), td, x, y, z, AA)

        Returns
        -------
        return a listed PDKit gait features as dictionary
        """

        # check if dataframe is valid
        window_start = accel_dataframe.td[0]
        window_end = accel_dataframe.td[-1]
        window_duration = window_end - window_start
        feature_dict = {}
        for axis in ["x", "y", "z", "AA"]:
            gp = pdkit.GaitProcessor(
                duration = window_duration,
                cutoff_frequency=self.pdkit_gait_features_cutoff_frequency,
                filter_order=self.pdkit_gait_features_filter_order,
                delta=self.pdkit_gait_features_delta,
                sampling_frequency=self.sensor_sampling_frequency,
                lower_frequency=self.pdkit_gait_features_low_freq,
                upper_frequency =self.pdkit_gait_features_upper_freq,
                step_size=self.pdkit_gait_features_step_size,
                stride_fraction = self.pdkit_gait_features_stride_fraction)
            series = accel_dataframe[axis]
            var = series.var()
            try:
                if (var) < self.sensor_variance_cutoff:
                    steps = 0
                    cadence = 0
                else:
                    strikes, _ = gp.heel_strikes(series)
                    steps = np.size(strikes)
                    cadence = steps/window_duration
            except (IndexError, ValueError):
                steps = 0
                cadence = 0

            if steps >= 2:
                step_durations = []
                for i in range(1, np.size(strikes)):
                    step_durations.append(strikes[i] - strikes[i-1])
                avg_step_duration = np.mean(step_durations)
                sd_step_duration = np.std(step_durations)
            else:
                avg_step_duration = np.NaN
                sd_step_duration = np.NaN

            if (steps >= 4) and \
                    (avg_step_duration > 1/self.sensor_sampling_frequency):
                strides1 = strikes[0::2]
                strides2 = strikes[1::2]
                stride_durations1 = []
                for i in range(1, np.size(strides1)):
                    stride_durations1.append(strides1[i] - strides1[i-1])
                stride_durations2 = []
                for i in range(1, np.size(strides2)):
                    stride_durations2.append(strides2[i] - strides2[i-1])
                avg_number_of_strides = np.mean(
                    [np.size(strides1), np.size(strides2)])
                avg_stride_duration = np.mean((np.mean(stride_durations1),
                                               np.mean(stride_durations2)))
                sd_stride_duration = np.mean((np.std(stride_durations1),
                                              np.std(stride_durations2)))

                step_regularity, stride_regularity,\
                    symmetry = gp.gait_regularity_symmetry(
                        series,
                        average_step_duration=avg_step_duration,
                        average_stride_duration=avg_stride_duration)
            else:
                avg_number_of_strides = np.NaN
                avg_stride_duration = np.NaN
                sd_stride_duration = np.NaN
                step_regularity = np.NaN
                stride_regularity = np.NaN
                symmetry = np.NaN

            speed_of_gait = gp.speed_of_gait(series, wavelet_level=6)
            energy_freeze_index, \
                locomotor_freeze_index = self.get_freeze_index_features(
                    series, self.sensor_sampling_frequency)
            feature_dict["%s_energy_freeze_index" % axis] = energy_freeze_index
            feature_dict["%s_loco_freeze_index" %
                         axis] = locomotor_freeze_index
            feature_dict["%s_avg_step_duration" % axis] = avg_step_duration
            feature_dict["%s_sd_step_duration" % axis] = sd_step_duration
            feature_dict["%s_cadence" % axis] = cadence
            feature_dict["%s_avg_number_of_strides" %
                         axis] = avg_number_of_strides
            feature_dict["%s_avg_stride_duration" % axis] = avg_stride_duration
            feature_dict["%s_sd_stride_duration" % axis] = sd_stride_duration
            feature_dict["%s_speed_of_gait" % axis] = speed_of_gait
            feature_dict["%s_step_regularity" % axis] = step_regularity
            feature_dict["%s_stride_regularity" % axis] = stride_regularity
            feature_dict["%s_symmetry" % axis] = symmetry
        feature_dict["window_size"] = window_duration
        feature_dict["window_start"] = window_start
        feature_dict["window_end"] = window_end
        return feature_dict

    
    def detect_rotation_timepoint(self, gyro_dataframe, axis):
        """
        Function to output rotation information (omega, period of rotation)
        in a dictionary format, given a gyroscope dataframe
        and its axis orientation
        
        Parameters
        -----------
        gyro_dataframe (type: pd.DataFrame): A gyrosocope dataframe
        axis (type: str, default = "y"): Rotation-rate axis

        Returns
        --------
            Returns a dictionary of omega and period of rotaton
            Return Format: {"chunk_1": {omega: value, period: value},
                            "chunk_2": {omega: value, period: value}}
        """
        gyro_dataframe[axis] = butter_lowpass_filter(
            data=gyro_dataframe[axis],
            sample_rate=self.sensor_sampling_frequency,
            cutoff=self.rotation_detection_frequency_cutoff,
            order=self.rotation_detection_filter_order)
        zero_crossings = np.where(np.diff(np.sign(gyro_dataframe[axis])))[0]
        start = 0
        num_rotation_window = 0
        rotation_dict = {}
        for crossing in zero_crossings:
            if (gyro_dataframe.shape[0] >= 2) & (start != crossing):
                duration = gyro_dataframe["td"][crossing] - \
                    gyro_dataframe["td"][start]
                auc = np.abs(metrics.auc(
                    gyro_dataframe["td"][start: crossing + 1],
                    gyro_dataframe[axis][start: crossing + 1]))
                omega = auc/duration
                aucXt = auc * duration

                if aucXt > self.rotation_detection_aucXt_upper_limit:
                    num_rotation_window += 1
                    rotation_dict["rotation_chunk_%s" % num_rotation_window] =\
                        ({"omega": omega,
                          "aucXt": aucXt,
                          "duration": duration,
                          "period": [start, crossing]})
                start = crossing
        return rotation_dict

    def segment_gait_sequence(self, accel_dataframe, periods):
        """
        Function to chunk dataframe into several periods of rotational
        and non-rotational sequences into List.
        Each dataframe chunk in the list will be in dictionary format
        containing information of the dataframe itself
        and what type of chunk it categorizes as.

        Parameters
        -----------
        accel_dataframe (type: pd.Dataframe): User acceleration dataframe
        periods         (type: List): A list of list of periods
        periods input format: 
            [[start_ix, end_ix], [start_ix, end_ix]]

        Parameters
        -----------
        Returns a List of dictionary that has keys containing information
        of what type of chunk, and the dataframe itself.
        Return Format: [{chunk1 : walk_chunk_1, dataframe : data},
                        {chunk2 : rotation_chunk_1, dataframe : data}]
        """
        # instantiate initial values
        chunk_list = []
        pointer = 0
        num_rotation_window = 1
        num_walk_window = 1

        # check if there is rotation
        if len(periods) == 0:
            chunk_list.append(
                {"dataframe": accel_dataframe,
                 "chunk": "walk_chunk_%s" % num_walk_window})
            return chunk_list

        # edge case: if rotation occurs at zero
        if periods[0][0] == pointer:
            chunk_list.append(
                {"dataframe": accel_dataframe.iloc[pointer: periods[0][1] + 1],
                 "chunk": "rotation_chunk_%s" % num_rotation_window})
            num_rotation_window += 1
            pointer = periods[0][1] + 1
            periods = periods[1:]

        # middle case
        for index in periods:
            chunk_list.append(
                {"dataframe": accel_dataframe.iloc[pointer: index[0] + 1],
                 "chunk": "walk_chunk_%s" % num_walk_window})
            num_walk_window += 1
            chunk_list.append(
                {"dataframe": accel_dataframe.iloc[index[0] + 1: index[1] + 1],
                 "chunk": "rotation_chunk_%s" % num_rotation_window})
            num_rotation_window += 1
            pointer = index[1] + 1

        # edge case, last bit of data
        if pointer < accel_dataframe.shape[0]:
            chunk_list.append(
                {"dataframe": accel_dataframe.iloc[pointer:],
                 "chunk": "walk_chunk_%s" % num_walk_window})
        return chunk_list

    def featurize_data_segments(self,list_of_dataframe_dicts, rotation_dict):
        """
        Function to featurize list of rotation-segmented dataframe chunks with
        moving windows, and returns list of dictionary with all the pdkit
        and rotation features from each moving window

        Notes:
          1. if dataframe chunk is smaller than the window size or is a
          rotation dataframe chunk it will just be treated as one window
          2. if current dataframe size is bigger than window size,
          then iterate through dataframe chunk
        
        Parameters
        -----------
        list_of_dataframe_dicts (dtype = List): A List filled with
                                                dictionaries containing
                                                key-value pair of the
                                                dataframe and
                                                its chunk type
        rotation_dict (dtype = Dict): A Dictionary containing
                                      rotation information
                                      on each rotation_chunk.
        Returns
        -----------
            RType = List
            Returns a list of features from each window,
            each window features is stored as dictionary
            inside the list

            Return format:
                [{window1_features:...},
                {window2_features:...},
                {window3_features:...}]
        """

        feature_list = []
        num_window = 1
        step_size = int(self.sensor_window_size * self.sensor_window_overlap)
        # separate to rotation and non rotation
        for dataframe_dict in list_of_dataframe_dicts:
            end_window_pointer = self.sensor_window_size
            curr_chunk = dataframe_dict["chunk"]
            curr_dataframe = dataframe_dict["dataframe"]

            if curr_dataframe.shape[0] < 0.5 * self.sensor_sampling_frequency:
                continue

            # define if chunk is a rotation sequence
            if "rotation_chunk" in curr_chunk:
                rotation_omega = rotation_dict[curr_chunk]["omega"]
            else:
                rotation_omega = np.NaN

            # edge case where dataframe is smaller than window, or rotation
            if ((curr_dataframe.shape[0] < end_window_pointer)
                    or (rotation_omega > 0)):
                feature_dict = self.get_pdkit_gait_features(curr_dataframe)
                feature_dict["rotation_omega"] = rotation_omega
                feature_dict["window"] = "window_%s" % num_window
                feature_list.append(feature_dict)
                num_window += 1
                continue

            # ideal case when data chunk is larger than window ##
            while end_window_pointer < curr_dataframe.shape[0]:
                start_window_pointer = end_window_pointer - self.sensor_window_size
                subset = curr_dataframe.iloc[start_window_pointer:
                                             end_window_pointer]
                feature_dict = self.get_pdkit_gait_features(subset)
                feature_dict["rotation_omega"] = rotation_omega
                feature_dict["window"] = "window_%s" % num_window
                feature_list.append(feature_dict)
                end_window_pointer += step_size
                num_window += 1
        return feature_list


    def run_pipeline(self, accel_data, rotation_data):
        """
        main entry point of this featurizaton class, parameter will take in
        pd.DataFrame or the filepath to the dataframe.

        Parameters
        ----------
            accel_data (type: pd.DataFrame): accelerometer sensor time series 
            rotation_data (type: pd.DataFrame): rotation rate sensor time series

        Returns
        -------
            A list of dictionary. With each dictionary representing
            gait features on each window.
        """
        error = []
        rotation_dict = {}
        result = pd.DataFrame()
        try:
            resampled_accel = self.format_time_series(accel_data)
            resampled_rotation = self.format_time_series(rotation_data)
            if isinstance(resampled_accel, str):
                error.append(resampled_accel)
            if isinstance(resampled_rotation, str):
                error.append(resampled_rotation)
            if len(error) != 0:
                result["error"] = ["; ".join(error)]
                return result
            if self.detect_rotation == True:
                rotation_dict = self.detect_rotation_timepoint(resampled_rotation, 
                                                               self.rotation_detection_axis)
            periods = [v["period"] for k, v in rotation_dict.items()]
            list_df_chunks = self.segment_gait_sequence(resampled_accel, periods)
            list_of_feature_dict = self.featurize_data_segments(list_df_chunks, rotation_dict)
            result = pd.DataFrame(list_of_feature_dict)
            result["error"] = np.nan
            return result
        except Exception as err:
            result["error"] = [err]
            return result