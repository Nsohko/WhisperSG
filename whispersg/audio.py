import os
import subprocess
from functools import lru_cache
from typing import Optional, Union
from threading import Thread, Lock
import audioop

import numpy as np
import torch
import torch.nn.functional as F
import time

import pyaudio
import wave

from .utils import exact_div

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
CHANNELS = 1
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # Launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI to be installed.
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads",
            "0",
            "-i",
            file,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sr),
            "-",
        ]
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

# class that creates a new thread and waits for user input
# once user input is received, it will perform fn() and close.
# this will be used to terminate user recording
class InputThread(Thread):

    def __init__(self, fn = None, name='input_thread'):
        self.fn = fn
        super().__init__(name=name)

    # override run method
    def run(self):
        # wait for input
        x = input()

        # once input is recived, execute fn
        self.fn()

        # close thread
        return


class ArrayRecorder:

    def __init__(self, mic_id=1, dataformat=pyaudio.paInt16, channels=CHANNELS, sample_rate=SAMPLE_RATE):
        # this will be our recorder that will store data in an internal buffer

        # our main buffer that we will read data into. This will be a list of lists of bytes
        self.audio_buffer = []

        self.mic_id = mic_id

        self.pa = pyaudio.PyAudio()

        self.dataformat = dataformat
        self.channels = channels # note that al
        self.sample_rate = sample_rate
        self.sample_size = self.pa.get_sample_size(self.dataformat)

        self.active = False

    # starts recording audio from user and writes
    def listen(self):

        if not self.active:
            # start listening
            print('Initialising... Enter any key to quit')
            print(f"Using mic {self.mic_id}: {self.pa.get_device_info_by_index(self.mic_id).get('name')}")
            self.start_recorder()

            # wait for input, then stop recorder
            x=input()
            self.stop_recorder()

        else:
            print("Please terminate current activity first.")

    def start_recorder(self): # initiates recording

        def callback(in_data, frame_count, time_info, status):
            # simply put data into our main buffer when read
            self.audio_buffer.append(in_data)
            return (in_data, pyaudio.paContinue)

        # now that we have defined our callback fn, lets start the stream
        self.stream = self.pa.open(format=self.dataformat,
                                   channels=self.channels,
                                   rate=self.sample_rate,
                                   input=True,
                                   stream_callback=callback,
                                   input_device_index=self.mic_id,
                                   frames_per_buffer=1024
                                  )

        self.stream.start_stream()
        self.active = True
        time.sleep(0.5)
        print('Listening...')

    def stop_recorder(self): # terminates recording
        # if we are active:
        if self.active:
            self.stream.stop_stream()
            self.stream.close()

            self.active = False
            print('Recording Finished')

    def get_audio(self):  # gets output converted to float32 dtype numpy array (for direct integration with whisperx)
        time.sleep(1.0)
        raw_data = b''.join(self.audio_buffer)  # get the complete list
        return np.frombuffer(raw_data, np.int16).flatten().astype(np.float32) / 32768.0


class LiveArrayRecorder(ArrayRecorder):

    def __init__(self, mic_id=1, dataformat=pyaudio.paInt16, channels=CHANNELS, sample_rate=SAMPLE_RATE, silence_threshold=None):

        super().__init__(mic_id=mic_id, dataformat=dataformat, channels=channels, sample_rate=sample_rate)

        # a secondary buffer used to manage concurrent tasks (in the case of wait=False). This will also be a list of lists of bytes
        self.extra_buffer = []

        # a third buffer used by the 'check_new_data()' method. This is used to store a 'previous' state of the main buffer to check if there is any update
        # this will just a list of bytes
        self.previous_audio_buffer = []

        self.lock = Lock()  # a lock used to manage concurrent tasks (in the case that wait=False)

        self.silence_threshold = silence_threshold
        if self.silence_threshold is None:
            self.adjust_for_ambient_sound()



    # starts recording audio from user and writes
    def listen(self):

        if not self.active:
            # start listening
            print('Initialising... Enter any key to quit')
            print(f"Using mic {self.mic_id}: {self.pa.get_device_info_by_index(self.mic_id).get('name')}")
            self.start_recorder()
            time.sleep(0.5)
            print('Listening...')

            # start a new thread that will wait for input
            # once input is received it will stop the recorder
            InputThread( fn = self.stop_recorder ).start()

        else:
            print("Please terminate current activity first.")

    def start_recorder(self): # initiates recording

        # we are not waiting, so we need to use our lock to manage concurrent access to our buffers
        def callback(in_data, frame_count, time_info, status):
            # try to acquire lock in non-blocking mode. If it returns True we have acquired it; if it returns False we have not acquired it (as another task holds it)
            # note that in this case, if it returns True, it automatically acquires the lock

            # case 1: we acquire the lock
            if self.lock.acquire(blocking=False):
                # first update with data from extra_buffer
                self.audio_buffer.extend(self.extra_buffer)
                # then update with new data
                self.audio_buffer.append(in_data)
                # clear extra buffer
                self.extra_buffer.clear()
                # release lock
                self.lock.release()
            # case 2: we didn't acquire lock
            else:
                # since we didnt get lock, its not safe to write to our main data buffer, so we will instead temporarily store our data in our extra buffer
                self.extra_buffer.append(in_data)

            return (in_data, pyaudio.paContinue)

        # now that we have defined our callback fn, lets start the stream
        self.stream = self.pa.open(format=self.dataformat,
                                   channels=self.channels,
                                   rate=self.sample_rate,
                                   input=True,
                                   stream_callback=callback,
                                   input_device_index=self.mic_id,
                                   frames_per_buffer=1024
                                   )

        self.stream.start_stream()
        self.active = True

    def stop_recorder(self): # terminates recording
        # if we are active:
        if self.active:
            self.stream.stop_stream()
            self.stream.close()

            self.active = False

    # gets output as a NumPy array containing the audio waveform, in float32 dtype.
    def format_audio(self, raw_data):
        return np.frombuffer(raw_data, np.int16).flatten().astype(np.float32) / 32768.0

    def get_audio(self): # gets output converted to float32 dtype numpy array (for direct integration with whisperx)
        self.lock.acquire() # get lock
        raw_data = b''.join(self.audio_buffer) # get the complete list
        self.lock.release() # release lock

        return self.format_audio(raw_data)

    def get_audio_duration(self): # gets the length of audio stored in our data buffer

        self.lock.acquire()
        raw_data = b''.join(self.audio_buffer)  # get the complete list
        self.lock.release()

        total_frames = len(raw_data) / self.channels # get the total number of frames.
        total_samples = total_frames / self.sample_size # get the number of samples as number of frames divided by number of bytes per fram
        duration = total_samples / self.sample_rate # calculate duration

        return duration

    # empties our buffer
    def clear_data(self):

        self.lock.acquire()  # get lock
        raw_data = b''.join(self.audio_buffer)  # get a snapshot of current data
        self.audio_buffer.clear()  # empty our current data. Note that we cant use .clear() here since that would empty cur_data too
        self.lock.release()  # release our lock

        return self.format_audio(raw_data)

    def clear_if_silent(self, silence_duration=0.5):
        # get the number of bytes corresponding to the silence duration
        silence_bytes = int( silence_duration * self.sample_rate * self.sample_size * self.channels )
        silent = False

        self.lock.acquire()
        raw_data = b''.join(self.audio_buffer)
        audio_slice = raw_data[-silence_bytes:]

        if self.calculate_rms(audio_slice) < self.silence_threshold:
            silent = True
            self.audio_buffer.clear()

        self.lock.release()

        return self.format_audio(raw_data), silent

    def check_for_update(self): # checks whether there has been an update since the last time this function was called
        prev_len = len(self.previous_audio_buffer)

        update = True

        if self.lock.acquire(blocking=False):

            if len(self.audio_buffer) > prev_len:
                audio_slice = self.audio_buffer[-prev_len: ]

                if self.calculate_rms(audio_slice) < self.silence_threshold:
                    update = False

            self.previous_audio_buffer = self.audio_buffer.copy()
            self.lock.release()

        return update

    def calculate_rms(self, audio_slice):
        # Calculate RMS value
        rms = audioop.rms(audio_slice, self.sample_size)

        return rms

    def adjust_for_ambient_sound(self):
        print("\nAdjusting for ambient sound. Please stay silent during this period")
        self.start_recorder()
        time.sleep(2.0)
        self.stop_recorder()
        raw_audio = b''.join(self.audio_buffer)
        self.audio_buffer.clear()
        self.extra_buffer.clear()

        self.silence_threshold = self.calculate_rms(raw_audio)
        print(self.silence_threshold)

        print("Done adjusting\n")


# subclass if we are recording to a file
class FileRecorder:
    # inherit __init__ from recorder parent class
    def __init__(self, file_path, mic_id=1, dataformat=pyaudio.paInt16, channels=CHANNELS, sample_rate=SAMPLE_RATE):
        # this will be our recorder that will store data in a file
        self.file_path = file_path

        # our main buffer that we will read data into
        self.data = []

        self.mic_id = mic_id

        self.dataformat = dataformat
        self.channels = channels  # note that al
        self.sample_rate = sample_rate

        self.pa = pyaudio.PyAudio()

        # whether we are currenlt listening
        self.active = False

    # starts recording audio from user and writes
    def listen(self):

        if not self.active:
            # start listening
            print('Initialising... Enter any key to quit')
            print(f"Using mic {self.mic_id}: {self.pa.get_device_info_by_index(self.mic_id).get('name')}")
            self.start_recorder()

            x = input()
            self.stop_recorder()

        else:
            print("Please terminate current activity first.")

    # start recording audio
    def start_recorder(self):
        # open and instantiate wavfile we will write to
        self.wf = wave.open(self.file_path, 'wb')
        self.wf.setnchannels(self.channels)
        self.wf.setsampwidth(self.pa.get_sample_size(self.dataformat))
        self.wf.setframerate(self.sample_rate)

        # this is the fn that will be called whenever we have new data available
        def callback(in_data, frame_count, time_info, status):
            # file write should be able to keep up with audio data stream (about 1378 Kbps)
            self.wf.writeframes(in_data)
            return (in_data, pyaudio.paContinue)

        # create and start the audio stream
        self.stream = self.pa.open(format=self.dataformat,
                                    channels=self.channels,
                                    rate=self.sample_rate,
                                    input=True,
                                    stream_callback=callback,
                                    input_device_index=self.mic_id)

        self.stream.start_stream()
        self.active = True
        time.sleep(0.5)
        print('Listening...')

    # stops recording and closes files
    def stop_recorder(self):
        # close recording
        if self.active:
            self.stream.stop_stream()
            self.stream.close()
            self.wf.close()

            self.active = False
            print('Recording Finished')

# factory function to get the appropriate type of recorder (FileRecorder or ArrayRecorder) based on output_type
def get_recorder(file_path=None, mic_id=1, dataformat=pyaudio.paInt16, channels=CHANNELS, sample_rate=SAMPLE_RATE, wait=True, silence_threshold=None):
    # check if we were passed a str with a file path, if so lets use the FileRecorder subclass to write to a file
    if isinstance(file_path,str):
        dir_name = os.path.dirname(file_path)
        if not (os.path.isdir(dir_name)):
            raise ValueError(
                f"Invalid path provided; To create wav file in same directory, pass /[filename]; To write to Recorder's internal buffer, do not pass file_path parameter")
        if not wait:
            raise ValueError(
                "Non-blocking mode (wait=False) is not allowed when writing to a file; To use non-blocking mode, do not pass any parameter for file to write to recorder's internal buffer")

        return FileRecorder(file_path, mic_id, dataformat, channels, sample_rate)

    # otherwise if file_path is None, lets use our internal buffer instead
    elif file_path is None:
        if wait:
            return ArrayRecorder(mic_id, dataformat, channels, sample_rate)
        else:
            return LiveArrayRecorder(mic_id, dataformat, channels, sample_rate, silence_threshold)

    else:
        raise ValueError(
            "Invalid file path passed; To record into Recorder's internal buffer, leave the file_path parameter empty; To record into a file, pass a valif file path as a string")



def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels in [80, 128], f"Unsupported n_mels: {n_mels}"
    with np.load(
        os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    ) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec
