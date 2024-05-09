import argparse
import time
import os


from whispersg.audio import get_recorder
from whispersg.pipeline import WhisperSGPipeline
from whispersg.asr import DEFAULT_ASR_MODEL_PATH
from whispersg.utils import LANGUAGES, TO_LANGUAGE_CODE


import pyaudio


def cli():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--model", default=DEFAULT_ASR_MODEL_PATH, help="Name of the Whisper model to use (e.g. tiny / tiny.en / small / etc.; If using a local model, provide the path to the model folder instead; Default is the finetuned small.en model built in this project")
    parser.add_argument("--mic_id", default=pyaudio.PyAudio().get_default_input_device_info()["index"], help="The id of the computer mic to use; Use with \"list\" to list all available mics")
    parser.add_argument("--compute_type", default="float16", type=str, choices=["float16", "float32", "int8"], help="Compute type for computation")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="Whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default=None, choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]), help="Language spoken in the audio, specify None to perform language detection")

    parser.add_argument("--hard_chunk_limit", type=int, default=100, help="Maximum duration of a chunk. Immediately resets internal audio buffer once length of the buffer exceeds this value; Set as <= 0 to set maximum duration to infinity")
    parser.add_argument("--soft_chunk_limit", type=int, default=15, help="After length of audio buffer exceeds the soft_chunk limit, audio buffer will be cleared on next detection of silence; Set as <0 to disable feature")
    parser.add_argument("--silence_duration", type=int, default=0.1, help="Minimum duration of silence to be detecetd for soft_chunk_limit")
    parser.add_argument("--silence_threshold", type=int, default=None, help="RMS value of sounds corresponding to silence; Leave empty to automatically pick rms back on background sounds")


    args = parser.parse_args().__dict__

    asr_model_path = args["model"]

    mic_id = args["mic_id"]
    if mic_id == "list":
        p = pyaudio.PyAudio()
        num_devices = p.get_device_count()

        for i in range(num_devices):
            # Get the device info
            device_info = p.get_device_info_by_index(i)
            # Check if this device is a microphone (an input device)
            if device_info.get('maxInputChannels') > 0:
                print(f"Microphone: {device_info.get('index')} , Device Index: {device_info.get('name')}")
        return

    else:
        try:
            mic_id = int(mic_id)
        except:
            raise ValueError("Invalid mic id")

    compute_type = args["compute_type"]

    task = args["task"]
    language = args["language"]

    hard_chunk_limit = args["hard_chunk_limit"]
    soft_chunk_limit = args["soft_chunk_limit"]

    silence_threshold = args["silence_threshold"]
    silence_duration = args["silence_duration"]


    do_hard_chunk_limit = True
    do_soft_chunk_limit = True

    if hard_chunk_limit <= 0:
        do_hard_chunk_limit = False

    if soft_chunk_limit < 0:
        do_soft_chunk_limit = False

    # load model and non-blocking recorder
    transcriber = WhisperSGPipeline(whisper_model_name=asr_model_path, task=task, language=language, compute_type=compute_type, alignment=False, diarization=False)
    recorder = get_recorder(file_path=None, mic_id=mic_id, wait=False, silence_threshold=silence_threshold)

    transcriptions = [None]

    recorder.listen()
    while recorder.active:

        time.sleep(1.0)

        new_chunk = False # whether this is a new chunk or not

        audio_duration = recorder.get_audio_duration()

        if do_hard_chunk_limit and audio_duration > hard_chunk_limit:

            audio_np = recorder.clear_data()
            recorder.silence_threshold += 30
            new_chunk = True

        elif do_soft_chunk_limit and audio_duration > soft_chunk_limit:

            audio_np, new_chunk = recorder.clear_if_silent(silence_duration)

        else:
            audio_np = recorder.get_audio()

        if len(audio_np) == 0:
            continue

        transcription = transcriber.transcribe(audio_np)


        if len(transcription) == 0:
            continue

        transcriptions[-1] = transcription
        if new_chunk:
            transcriptions.append(None)

        os.system('cls' if os.name == 'nt' else 'clear')
        for transcription in transcriptions:
            if transcription is None:
                continue
            transcriber.print_transcription(transcription, show_timestamps=False)
            print('\n')

        print('...')

if __name__ == "__main__":
    cli()