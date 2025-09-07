import numpy as np
from scipy.io import savemat, loadmat
from scipy.interpolate import interp1d
import scipy.signal as signal
import h5py


def get_data(path, user, texture, trial, time_offset=-1691745000):
    """
    Load data from a .mat file and return the time and values.

    Args:
        path (str): The path to the directory containing the data.
        user (str): The user ID.
        texture (str): The texture ID.
        trial (str): The trial ID.
        time_offset (int, optional): The time offset to apply. Defaults to -1691745000.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The time and values arrays. [time_kistler, kistler_data], [time_pos, pos_data], [time_ft_sensor, ft_sensor_data]
    """
    filename = user + "_" + texture + "_" + trial + ".h5"
    with h5py.File(path + "/" + user + "/" + texture + "/" + filename, "r") as f:
        time_kistler = f["time_kistler"][:] + time_offset
        kistler_data = f["kistler_data"][:]
        time_pos = f["time_pos"][:] + time_offset
        pos_data = f["pos_data"][:]
        time_ft_sensor = f["time_ft_sensor"][:] + time_offset
        ft_sensor_data = f["ft_sensor_data"][:]
    return (
        [time_kistler, kistler_data],
        [time_pos, pos_data],
        [time_ft_sensor, ft_sensor_data],
    )


def get_position_features(time, values):
    """
    Compute position, speed, and direction features from the given time and values.

    Args:
        time (np.ndarray): The time array.
        values (np.ndarray): The values array.

    Returns:
        Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: The position, speed, and direction features.
    """
    # resampling the data at a constant sampling rate
    # create a new time array
    new_time = np.arange(time[0, 0], time[-1, 0], 1 / 50)
    # interpolate the values
    new_values = np.zeros((new_time.shape[0], values.shape[1]))
    for k in range(values.shape[1]):
        new_values[:, k] = interp1d(
            time[:, 0],
            values[:, k],
            axis=0,
            bounds_error=False,
            fill_value=0,
            kind="linear",
        )(new_time)
    # replace the old time and values
    time = np.reshape(new_time, (new_time.shape[0], 1))
    values = new_values

    # resampling the data at a higher sampling rate using cubic interpolation
    Fs = 240  # sampling rate
    # create a new time array
    new_time = np.arange(time[0, 0], time[-1, 0], 1 / Fs)
    # interpolate the values
    new_values = np.zeros((new_time.shape[0], values.shape[1]))
    for k in range(values.shape[1]):
        new_values[:, k] = interp1d(
            time[:, 0],
            values[:, k],
            axis=0,
            bounds_error=False,
            fill_value=0,
            kind="cubic",
        )(new_time)
    # replace the old time and values
    time = np.reshape(new_time, (new_time.shape[0], 1))
    values = new_values

    # filter the data
    fc = 15  # Cut-off frequency of the filter
    a, b = signal.butter(10, fc / (Fs / 2), "low")
    for k in range(values.shape[1]):
        values[:, k] = signal.filtfilt(a, b, values[:, k])

    # filter the data
    posX = values[:, 0]
    posY = values[:, 1]
    posX2 = values[:, 2]
    posY2 = values[:, 3]
    step_distance = np.sqrt(np.diff(posX) ** 2 + np.diff(posY) ** 2)
    speed = step_distance / (np.diff(time[:, 0]))
    speed = np.append(speed, speed[-1])

    # distance = np.append(distance,distance[-1])
    position = np.concatenate(
        (posX.reshape(posX.shape[0], 1), posY.reshape(posY.shape[0], 1)), axis=1
    )

    # create an array of same size of values with the speed and direction of the movement
    direction = np.zeros((position.shape[0],))
    finger_direction = np.zeros((position.shape[0],))
    nb_turn = 0  # keep track of the number of turn
    for i in range(1, position.shape[0] - 1):
        direction[i] = (
            np.arctan2(posY[i + 1] - posY[i - 1], posX[i + 1] - posX[i - 1])
            * 180
            / np.pi
        )
        # unrap the direction
        if direction[i] - (direction[i - 1] - nb_turn * 360) < -320:
            nb_turn = nb_turn + 1
        elif direction[i] - (direction[i - 1] - nb_turn * 360) > 320:
            nb_turn = nb_turn - 1
        direction[i] = direction[i] + nb_turn * 360

        # check the distance between the two markers
        dist = np.sqrt((posX2[i] - posX[i]) ** 2 + (posY2[i] - posY[i]) ** 2)
        if (
            abs(0.150 - dist) < 0.04
        ):  # the distance between the two markers should be constant around 0.15
            finger_direction[i] = (
                np.arctan2(posY2[i] - posY[i], posX2[i] - posX[i]) * 180 / np.pi
                + 90
                - 20
            )  # 20 is the angle between the finger and the marker
        else:  # if the distance is not constant, the second marker is probably not well detected
            if i > 0:
                finger_direction[i] = finger_direction[i - 1]
    direction = direction - finger_direction  # correct the direction of the finger

    return [time, position, speed, direction]


def get_kistler_features(time, values):
    """
    Compute audio and vibration features from the given time and values.

    Args:
        time (np.ndarray): The time array.
        values (np.ndarray): The values array.

    Returns:
        Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: The audio and vibration features.
    """
    audio = values[:, 0:2]  # audio 1 and 2
    time_audio = time
    vibration = values[:, 2:4]  # vibration 1 and 2
    time_vibration = time
    return [time_audio, audio], [time_vibration, vibration]


def get_ft_sensor_features(time, values):
    """
    Compute force and torque features from the given time and values.

    Args:
        time (np.ndarray): The time array.
        values (np.ndarray): The values array.

    Returns:
        Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: The force and torque features.
    """
    force = values[:, 0:3]
    time_force = time
    torque = values[:, 3:6]
    time_torque = time
    return [time_force, force], [time_torque, torque]


def align_all(list_of_times, list_of_data, sampling_rate):
    """
    Align the given time and data arrays to a common time array.

    Args:
        list_of_times (List[np.ndarray]): A list of time arrays.
        list_of_data (List[np.ndarray]): A list of data arrays.
        sampling_rate (int): The sampling rate to use for the common time array.

    Returns:
        Tuple[np.ndarray, List[np.ndarray]]: The common time array and the aligned data arrays.
    """
    time_1d = [t.squeeze() for t in list_of_times]
    start_time = max([time[0] for time in list_of_times])[0]
    end_time = min([time[-1] for time in list_of_times])[0]
    common_time = np.arange(start_time, end_time, 1 / sampling_rate)
    aligned_data = [
        interp1d(time, data, axis=0)(common_time)
        for time, data in zip(time_1d, list_of_data)
    ]
    return common_time, aligned_data


# plot all position data on one plot offset by 0.1
def load_data(path, user, texture, trial, sampling_rate, time_offset=-1691745000):
    """
    Load data from the given path, user, texture, and trial.

    Args:
        path (str): The path to the directory containing the data.
        user (str): The user ID.
        texture (str): The texture ID.
        trial (str): The trial ID.
        sampling_rate (int): The sampling rate to use for the common time array.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The loaded data.
    """
    (
        [time_kistler, kistler_data],
        [time_pos, pos_data],
        [time_ft_sensor, ft_sensor_data],
    ) = get_data(path, user, texture, trial, time_offset)

    [time_pos, pos, spd, dir] = get_position_features(time_pos, pos_data)
    [time_force, force], [time_torque, torque] = get_ft_sensor_features(
        time_ft_sensor, ft_sensor_data
    )
    [time_audio, mic], [time_vibration, vib] = get_kistler_features(
        time_kistler, kistler_data
    )

    # store all data in lists
    times = [
        time_pos,
        time_pos,
        time_pos,
        time_vibration,
        time_audio,
        time_force,
        time_torque,
    ]
    datas = [pos, spd, dir, vib, mic, force, torque]

    time_min = min([time[0] for time in times])
    times = [time - time_min for time in times]

    common_time, aligned_data = align_all(times, datas, sampling_rate)

    time_pos = time_pos[:, 0] - time_pos[0, 0]
    common_time = common_time[:] - common_time[0]

    position = aligned_data[0]
    speed = aligned_data[1]
    direction = aligned_data[2]
    vibration = aligned_data[3]
    audio = aligned_data[4]
    force = aligned_data[5]
    torque = aligned_data[6]

    return [common_time, position, speed, direction, vibration, audio, force, torque], [
        time_pos,
        pos,
        spd,
        dir,
    ]
