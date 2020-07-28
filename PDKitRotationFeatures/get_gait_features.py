"""
Author: Sage Bionetworks

CP: Aryton Tediarjo (atediarjo@gmail.com)

About:
Featurization Pipeline Class on Gait Signal Data based on
userAcceleration, and rotation rate (gyroscope). This wrapper is based
on the python package PDKit which will be added with some extra
functionalities (windowing), rotation QC steps,
and compatibility in running with mpower data
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


class GaitFeaturize:

    """
    This is the gait featurization package that combines features from pdkit,
    rotation detection algorithm from gyroscope,
    using overlapped-moving window (based on mHealthTools)

    References:
        Rotation-Detection Paper : https://www.ncbi.nlm.nih.gov/pmc/\
                                    articles/PMC5811655/
        PDKit Docs               : https://pdkit.readthedocs.io/\
                                    _/downloads/en/latest/pdf/
        PDKit Gait Source Codes  : https://github.com/pdkit/gait_processor.py
        Freeze of Gait Docs      : https://ieeexplore.ieee.org/document/5325884

    Args:
        gyro_frequency_cutoff (dtype: int, float) : frequency cutoff for
                                                    gyroscope rotation rate
        gyro_filter_order (dtype: float)          : signal filter order for
                                                    gyroscope rotation rate
        gyro_aucXt_upper_limit (dtype: int, float): upper limit of AUC during
                                                    "zero crossing period",
                                                    an upper limit of above
                                                    two indicates that user is
                                                    doing a turn/rotation
                                                    movements (From Ref 1)
        accel_frequency_cutoff (dtype: int, float): Frequency cutoff on
                                                    user acceleration
                                                    (default = 5Hz)
        accel_filter_order (dtype: int, float)    : Signal filter order on
                                                    user acceleration
                                                    (default = 4th Order)
        accel_delta (dtype: int, float)           : A point is considered a
                                                    maximum peak if it has the
                                                    maximal value, and was
                                                    preceded (to the left)
                                                    by a value lower
                                                    by delta (0.5 default).
                                                    (From Ref 2)
        variance_cutoff (dtype: int, float)       : variance cutoff on signal
        window_size (dtype: int, float)           : window size
                                                    default: 512 (5.12 secs)
        window_overlap (dtype: float)             : overlapped
                                                    window percentage
                                                    (20% overlap)
        loco_band (dtype: list)                   : The ratio of the energy in
                                                    the locomotion band
                                                    measured in Hz
                                                    default: 0.5-3Hz
        freeze_band (dtype: list)                 : The ration of energy in
                                                    the freeze band
                                                    measured in Hz
                                                    default: 3-8Hz
        sampling_frequency (dtype: float, int)    : Samples per seconds
    """

    def __init__(self,
                 rotation_frequency_cutoff=2,
                 rotation_filter_order=2,
                 rotation_aucXt_upper_limit=2,
                 pdkit_gait_frequency_cutoff=5,
                 pdkit_gait_filter_order=4,
                 pdkit_gait_delta=0.5,
                 variance_cutoff=1e-4,
                 window_size=512,
                 window_overlap=0.2,
                 loco_band=[0.5, 3],
                 freeze_band=[3, 8],
                 sampling_frequency=100):
        self.rotation_frequency_cutoff = rotation_frequency_cutoff
        self.rotation_filter_order = rotation_filter_order
        self.rotation_aucXt_upper_limit = rotation_aucXt_upper_limit
        self.pdkit_gait_filter_order = pdkit_gait_filter_order
        self.pdkit_gait_frequency_cutoff = pdkit_gait_frequency_cutoff
        self.pdkit_gait_delta = pdkit_gait_delta
        self.variance_cutoff = variance_cutoff
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.loco_band = loco_band
        self.freeze_band = freeze_band
        self.sampling_frequency = sampling_frequency

    def format_time_series_data(self, data):
        """
        Utility function to format time series

        Args:
            data(type: pd.DataFrame): pandas dataframe of time series

        Returns:
            RType: pd.DataFrame
            Returns formatted dataframe

        Return Dataframe Columns:
            t (dtype: DatetimeIndex) : An index of time in seconds
            x  (dtype: float64) : sensor-values in x axis
            y  (dtype: float64) : sensor-values in y axis
            z  (dtype: float64) : sensor-values in z axis
            AA (dtype: float64) : resultant of sensor values
            td (dtype: float64) : current time difference from zero in seconds
        """
        data = data.dropna(subset=["x", "y", "z"])
        data["AA"] = np.sqrt(data["x"]**2 + data["y"]**2 + data["z"]**2)
        data["td"] = data["t"] - data["t"].iloc[0]
        data["t"] = pd.to_datetime(data["td"], unit="s")
        data = data.set_index("t")
        # sort index
        data = data.sort_index()
        # remove data duplicates
        data = data[~data.index.duplicated(keep='first')]
        return data[["td", "x", "y", "z", "AA"]]

    def format_mpower_ts(self, data):
        """
        Function to format mPower Dataframe after subselecting
        which sensor types it tries to query
        Args:
            data (dtype: dataframe): dataframe time-series
        Returns:
            returns a cleaned and resampled sensor dataframe
        """
        try:
            if data.empty:
                return "ERROR: Filepath has Empty Dataframe"
            elif not (set(data.columns) >= set(["x", "y", "z", "t"])):
                return "ERROR: [t, x, y, z] in columns is required"
            else:
                data = self.format_time_series_data(data)
            return self.resample_signal(data)
        except(TypeError, AttributeError) as err:
            return "ERROR: %s" % type(err).__name__

    def calculate_freeze_index(self, accel_series, sample_rate):
        """
        Modified pdkit FoG freeze index function to be compatible with
        current data gait pipeline
        Args:
            accel_series (dtype = pd.Series): pd.Series of acceleration signal
            in one axis
            sample_rate  (dtype = float64)  : signal sampling rate
        Returns:
            RType: List
            List containing 2 freeze index values

        Return Format:
            [energy of freeze index, loco freeze index]
        """

        try:
            loco_band = self.loco_band
            freeze_band = self.freeze_band
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

    def get_gait_rotation_info(self, gyro_dataframe, axis="y"):
        """
        Function to output rotation information (omega, period of rotation)
        in a dictionary format, given a gyroscope dataframe
        and its axis orientation

        Args:
            gyro_dataframe (type: pd.DataFrame): A gyrosocope dataframe
            axis (type: str, default = "y")    : Rotation-rate axis

        Returns:
            RType: Dictionary

        Return Format:
         {"chunk_1": {omega: value,
                        period: value},
            chunk_2": {omega: value,
                        period: value}}
        """
        gyro_dataframe[axis] = butter_lowpass_filter(
            data=gyro_dataframe[axis],
            sample_rate=self.sampling_frequency,
            cutoff=self.rotation_frequency_cutoff,
            order=self.rotation_filter_order)
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

                if aucXt > self.rotation_aucXt_upper_limit:
                    num_rotation_window += 1
                    rotation_dict["rotation_chunk_%s" % num_rotation_window] =\
                        ({"omega": omega,
                          "aucXt": aucXt,
                          "duration": duration,
                          "period": [start, crossing]})
                start = crossing
        return rotation_dict

    def split_gait_dataframe_to_chunks(self, accel_dataframe, periods):
        """
        Function to chunk dataframe into several periods of rotational
        and non-rotational sequences into List.
        Each dataframe chunk in the list will be in dictionary format
        containing information of the dataframe itself
        and what type of chunk it categorizes as.

        Args:
            accel_dataframe (type: pd.Dataframe) : User acceleration dataframe
            periods         (type: List)         : A list of list of periods
            periods input format:
                [[start_ix, end_ix], [start_ix, end_ix]]

        Returns:
            Rtype: List
            Returns a List of dictionary that has keys containing information
            of what type of chunk, and the dataframe itself.
            Return format:
                [{chunk1 : walk_chunk_1, dataframe : pd.Dataframe},
                 {chunk2 : rotation_chunk_1, dataframe : pd.Dataframe}]
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

    def featurize_gait_dataframe_chunks_by_window(self,
                                                  list_of_dataframe_dicts,
                                                  rotation_dict):
        """
        Function to featurize list of rotation-segmented dataframe chunks with
        moving windows, and returns list of dictionary with all the pdkit
        and rotation features from each moving window

        Notes:
          1. if dataframe chunk is smaller than the window size or is a
          rotation dataframe chunk it will just be treated as one window
          2. if current dataframe size is bigger than window size,
          then iterate through dataframe chunk

        Args:
            list_of_dataframe_dicts (dtype = List): A List filled with
                                                    dictionaries containing
                                                    key-value pair of the
                                                    dataframe and
                                                    its chunk type
            rotation_dict (dtype = Dict): A Dictionary containing
                                            rotation information
                                            on each rotation_chunk.

        Returns:
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
        step_size = int(self.window_size * self.window_overlap)
        # separate to rotation and non rotation
        for dataframe_dict in list_of_dataframe_dicts:
            end_window_pointer = self.window_size
            curr_chunk = dataframe_dict["chunk"]
            curr_dataframe = dataframe_dict["dataframe"]

            if curr_dataframe.shape[0] < 0.5 * self.sampling_frequency:
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
                start_window_pointer = end_window_pointer - self.window_size
                subset = curr_dataframe.iloc[start_window_pointer:
                                             end_window_pointer]
                feature_dict = self.get_pdkit_gait_features(subset)
                feature_dict["rotation_omega"] = rotation_omega
                feature_dict["window"] = "window_%s" % num_window
                feature_list.append(feature_dict)
                end_window_pointer += step_size
                num_window += 1
        return feature_list

    def get_pdkit_gait_features(self, accel_dataframe):
        """
        Function to featurize dataframe using pdkit package (gait)
        based on all axis orientation (x, y, z, AA).

        Features from pdkit contains the following:
            1. Number of steps
            2. Cadence
            3. Freeze Index
            4. Locomotor Freeze Index
            5. Average Step/Stride Duration
            6. Std of Step/Stride Duration
            7. Step/Stride Regularity
            8. Speed of Gait
            9. Symmetry

        Args:
            accel_dataframe (type = pd.DataFrame)

        Returns:
            RType: Dictionary
            Returns dictionary containing features
            computed from pdkit
        """

        # check if dataframe is valid
        window_start = accel_dataframe.td[0]
        window_end = accel_dataframe.td[-1]
        window_duration = window_end - window_start
        feature_dict = {}
        for axis in ["x", "y", "z", "AA"]:
            gp = pdkit.GaitProcessor(
                duration=window_duration,
                cutoff_frequency=self.pdkit_gait_frequency_cutoff,
                filter_order=self.pdkit_gait_filter_order,
                delta=self.pdkit_gait_delta,
                sampling_frequency=self.sampling_frequency)
            series = accel_dataframe[axis]
            var = series.var()
            try:
                if (var) < self.variance_cutoff:
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
                    (avg_step_duration > 1/self.sampling_frequency):
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
                locomotor_freeze_index = self.calculate_freeze_index(
                    series, self.sampling_frequency)
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

    def resample_signal(self, dataframe):
        """
        Utility method for data resampling,
        Data will be interpolated using linear method

        Args:
            dataframe: A time-indexed dataframe

        Returns:
            RType: pd.DataFrame
            Returns a resampled dataframe based on predefined sampling
            frequency
        """
        new_freq = np.round(1 / self.sampling_frequency, decimals=6)
        df_resampled = dataframe.resample(str(new_freq) + 'S').mean()
        df_resampled = df_resampled.interpolate(method='linear')
        return df_resampled

    def run_gait_feature_pipeline(self, accel_data, rotation_data):
        """
        main entry point of this featurizaton class, parameter will take in
        pd.DataFrame or the filepath to the dataframe.

        Args:
            data (dtype: dataframe): dataframe time-series

        Returns:
            Rtype: List
            A list of dictionary. With each dictionary representing
            gait features on each window.
        """
        error = []
        result = pd.DataFrame()
        try:
            resampled_accel = self.format_mpower_ts(accel_data)
            resampled_rotation = self.format_mpower_ts(rotation_data)
            if isinstance(resampled_accel, str):
                error.append(resampled_accel)
            if isinstance(resampled_rotation, str):
                error.append(resampled_rotation)
            if len(error) != 0:
                result["error"] = ["; ".join(error)]
                return result
            rotation_dict = self.get_gait_rotation_info(resampled_rotation)
            periods = [v["period"] for k, v in rotation_dict.items()]
            list_df_chunks = self.split_gait_dataframe_to_chunks(
                resampled_accel, periods)
            list_of_feature_dictionary = \
                self.featurize_gait_dataframe_chunks_by_window(
                    list_df_chunks, rotation_dict)
            result = pd.DataFrame(list_of_feature_dictionary)
            result["error"] = np.nan
            return result
        except Exception as err:
            result["error"] = [err]
            return result
