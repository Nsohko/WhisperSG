import argparse
import gc
import os
import warnings

import numpy as np
import torch

import pyaudio

from whispersg.alignment import align, load_align_model
from whispersg.asr import load_model, DEFAULT_ASR_MODEL_PATH
from whispersg.audio import load_audio, get_recorder
from whispersg.diarize import DiarizationPipeline, assign_word_speakers
from whispersg.utils import (LANGUAGES, TO_LANGUAGE_CODE, DEFAULT_OUTPUT_PATH, optional_float,
                    optional_int, str2bool)



def cli():
    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio", nargs="*", type=str, help="Audio file(s) to transcribe; Leave empty to record audio from computer mic instead")

    parser.add_argument("--mic_id", default=pyaudio.PyAudio().get_default_input_device_info()["index"], help="The id of the computer mic to use; Use with \"list\" to list all available mics")
    parser.add_argument("--save_recording", action="store_true", help="Save recorded audio to disc; Only applicable when recording audio from computer mic; Recommended for long recordings; Saves audio to same address as --output_dir")

    # path to the small.en model fine-tuned on the partial IMDA dataset. Used as default model
    parser.add_argument("--model", default=DEFAULT_ASR_MODEL_PATH, help="Name of the Whisper model to use (e.g. tiny / tiny.en / small / etc.; If using a local model, provide the path to the model folder instead; Default is the finetuned small.en model built in this project")
    parser.add_argument("--cache_dir", type=str, default=None, help="The path to save model files; Uses ~/.cache/whisper by default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for PyTorch inference")
    parser.add_argument("--device_index", default=0, type=int, help="Device index to use for FasterWhisper inference")
    parser.add_argument("--batch_size", default=8, type=int, help="The preferred batch size for inference")
    parser.add_argument("--compute_type", default="float16", type=str, choices=["float16", "float32", "int8"], help="Compute type for computation")

    parser.add_argument("--save_results", action="store_true", help="Save transcription results to disc in output_dir")
    parser.add_argument("--output_dir", "-o", type=str, default=DEFAULT_OUTPUT_PATH, help="Directory to save the outputs")
    parser.add_argument("--output_format", "-f", type=str, default="all", choices=["all", "srt", "vtt", "txt", "tsv", "json", "aud"], help="Format of the output file; if not specified, all available formats will be produced")
    parser.add_argument("--verbose", type=str2bool, default=True, help="Whether to print out the progress and debug messages")

    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="Whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default=None, choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]), help="Language spoken in the audio, specify None to perform language detection")

    # alignment params
    parser.add_argument("--align_model", default=None, help="Name of phoneme-level ASR model to do alignment")
    parser.add_argument("--interpolate_method", default="nearest", choices=["nearest", "linear", "ignore"], help="For word .srt, method to assign timestamps to non-aligned words, or merge them into neighbouring.")
    parser.add_argument("--no_align", action='store_true', help="Do not perform phoneme alignment")
    parser.add_argument("--return_char_alignments", action='store_true', help="Return character-level alignments in the output json file")

    # vad params
    parser.add_argument("--vad_onset", type=float, default=0.400, help="Onset threshold for VAD (see pyannote.audio), reduce this if speech is not being detected")
    parser.add_argument("--vad_offset", type=float, default=0.300, help="Offset threshold for VAD (see pyannote.audio), reduce this if speech is not being detected.")
    parser.add_argument("--chunk_size", type=int, default=30, help="Chunk size for merging VAD segments. Default is 30, reduce this if the chunk is too long.")

    # diarization params
    parser.add_argument("--diarize", action="store_true", help="Apply diarization to assign speaker labels to each segment/word")
    parser.add_argument("--min_speakers", default=None, type=int, help="Minimum number of speakers to in audio file")
    parser.add_argument("--max_speakers", default=None, type=int, help="Maximum number of speakers to in audio file")
    
    parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
    parser.add_argument("--best_of", type=optional_int, default=5, help="number of candidates when sampling with non-zero temperature")
    parser.add_argument("--beam_size", type=optional_int, default=5, help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=1.0, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

    parser.add_argument("--suppress_tokens", type=str, default="-1", help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    parser.add_argument("--suppress_numerals", action="store_true", help="whether to suppress numeric symbols and currency symbols during sampling, since wav2vec2 cannot align them correctly")

    parser.add_argument("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
    parser.add_argument("--condition_on_previous_text", type=str2bool, default=False, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")
    parser.add_argument("--fp16", type=str2bool, default=True, help="whether to perform inference in fp16; True by default")

    parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")

    parser.add_argument("--max_line_width", type=optional_int, default=None, help="(not possible with --no_align) the maximum number of characters in a line before breaking the line")
    parser.add_argument("--max_line_count", type=optional_int, default=None, help="(not possible with --no_align) the maximum number of lines in a segment")
    parser.add_argument("--highlight_words", type=str2bool, default=False, help="(not possible with --no_align) underline each word as it is spoken in srt and vtt")
    parser.add_argument("--segment_resolution", type=str, default="sentence", choices=["sentence", "chunk"], help="(not possible with --no_align) the maximum number of characters in a line before breaking the line")

    parser.add_argument("--threads", type=optional_int, default=0, help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")

    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face Access Token to access PyAnnote gated models")

    parser.add_argument("--print_progress", type=str2bool, default = False, help = "if True, progress will be printed in transcribe() and align() methods.")
    # fmt: on

    args = parser.parse_args().__dict__

    mic_id = args.pop("mic_id")
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
    save_recording = args.pop("save_recording")
    
    model_name: str = args.pop("model")
    batch_size: int = args.pop("batch_size")
    model_dir: str = args.pop("cache_dir")

    save_results: bool = args.pop("save_results")
    output_dir: str = args.pop("output_dir")
    output_format: str = args.pop("output_format")

    device: str = args.pop("device")
    device_index: int = args.pop("device_index")
    compute_type: str = args.pop("compute_type")

    # model_flush: bool = args.pop("model_flush")
    os.makedirs(output_dir, exist_ok=True)

    align_model: str = args.pop("align_model")
    interpolate_method: str = args.pop("interpolate_method")
    no_align: bool = args.pop("no_align")
    task: str = args.pop("task")
    if task == "translate":
        # translation cannot be aligned
        no_align = True
    if task == "translate" and model_name.endswith('.en'):
        raise ValueError("Translation can only be done with multilingual models (small, medium, etc.), not english-only models (e.g. small.en); Please pass --model small instead for instance")

    return_char_alignments: bool = args.pop("return_char_alignments")

    hf_token: str = args.pop("hf_token")
    vad_onset: float = args.pop("vad_onset")
    vad_offset: float = args.pop("vad_offset")

    chunk_size: int = args.pop("chunk_size")

    diarize: bool = args.pop("diarize")
    min_speakers: int = args.pop("min_speakers")
    max_speakers: int = args.pop("max_speakers")
    
    print_progress: bool = args.pop("print_progress")

    if args["language"] is not None:
        args["language"] = args["language"].lower()
        if args["language"] not in LANGUAGES:
            if args["language"] in TO_LANGUAGE_CODE:
                args["language"] = TO_LANGUAGE_CODE[args["language"]]
            else:
                raise ValueError(f"Unsupported language: {args['language']}")

    if model_name.endswith(".en") and args["language"] != "en":
        if args["language"] is not None:
            warnings.warn(
                f"{model_name} is an English-only model but received '{args['language']}'; using English instead."
            )
        args["language"] = "en"
    align_language = args["language"] if args["language"] is not None else "en" # default to loading english if not specified

    temperature = args.pop("temperature")
    if (increment := args.pop("temperature_increment_on_fallback")) is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = [temperature]

    faster_whisper_threads = 4
    if (threads := args.pop("threads")) > 0:
        torch.set_num_threads(threads)
        faster_whisper_threads = threads

    asr_options = {
        "beam_size": args.pop("beam_size"),
        "patience": args.pop("patience"),
        "length_penalty": args.pop("length_penalty"),
        "temperatures": temperature,
        "compression_ratio_threshold": args.pop("compression_ratio_threshold"),
        "log_prob_threshold": args.pop("logprob_threshold"),
        "no_speech_threshold": args.pop("no_speech_threshold"),
        "condition_on_previous_text": False,
        "initial_prompt": args.pop("initial_prompt"),
        "suppress_tokens": [int(x) for x in args.pop("suppress_tokens").split(",")],
        "suppress_numerals": args.pop("suppress_numerals"),
    }

    writer = get_writer(output_format, output_dir)
    terminal_writer = get_writer("vtt", None)
    
    word_options = ["highlight_words", "max_line_count", "max_line_width"]
    if no_align:
        for option in word_options:
            if args[option]:
                parser.error(f"--{option} not possible with --no_align")
    if args["max_line_count"] and not args["max_line_width"]:
        warnings.warn("--max_line_count has no effect without --max_line_width")
    writer_args = {arg: args.pop(arg) for arg in word_options}
    
    # Part 1: VAD & ASR Loop
    results = []
    tmp_results = []
    # model = load_model(model_name, device=device, download_root=model_dir)
    model = load_model(model_name, device=device, device_index=device_index, download_root=model_dir, compute_type=compute_type, language=args['language'], asr_options=asr_options, vad_options={"vad_onset": vad_onset, "vad_offset": vad_offset}, task=task, threads=faster_whisper_threads)

    # get audio paths
    audio_list = args.pop("audio")

    # if at least 1 audio path was provided, we will transcribe the audio file
    if len(audio_list) >= 1:
        for audio_path in audio_list:
            audio = load_audio(audio_path)
            # >> VAD & ASR
            print(">>Performing transcription...")
            result = model.transcribe(audio, batch_size=batch_size, chunk_size=chunk_size, print_progress=print_progress)
            results.append((result, audio_path))
    # if no audio path was provided, default to recording with microphone
    else:
        # create audio path
        audio_path = os.path.join(output_dir, "whispersg_recording.wav")

        # if we are saving audio to disc
        if save_recording:
            # initiate a listener class with wait=True. Reads into audio_path
            recorder = get_recorder(file_path=audio_path, mic_id=mic_id, wait=True)
            # start listening, wait for user to enter any key to stop
            recorder.listen()
            # once done, load the audio back into memory
            audio = load_audio(audio_path)

        # if we are not saving to disc, we will kep audio in a buffer in memory
        else:
            # same as above, except read audio data into l's internal buffer
            recorder = get_recorder(mic_id=mic_id, wait=True)
            recorder.listen()
            # once done, load the audio from internal buffer
            audio = recorder.get_audio()
            # note that we still need "audio_path" for the writers later to know where to write output transcriptions files
            # however, when keeping audio data in ram, this is just a dummy path since no data is actually written there

        print(">>Performing transcription...")
        result = model.transcribe(audio, batch_size=batch_size, chunk_size=chunk_size, print_progress=print_progress)
        results.append((result, audio_path))


    # Unload Whisper, VAD and recorder
    del model
    gc.collect()
    torch.cuda.empty_cache()


    # Part 2: Align Loop
    if not no_align:
        tmp_results = results
        results = []
        align_model, align_metadata = load_align_model(align_language, device, model_name=align_model)
        for result, audio_path in tmp_results:
            # >> Align
            if len(tmp_results) > 1:
                input_audio = audio_path
            else:
                # lazily load audio from part 1. Also works when recording from mic into ram (i.e without saving to disc)
                # lets us ignore audio_path when keeping in ram, since it will access the numpy array directly instead
                input_audio = audio

            if align_model is not None and len(result["segments"]) > 0:
                if result.get("language", "en") != align_metadata["language"]:
                    # load new language
                    print(f"New language found ({result['language']})! Previous was ({align_metadata['language']}), loading new alignment model for new language...")
                    align_model, align_metadata = load_align_model(result["language"], device)
                print(">>Performing alignment...")
                result = align(result["segments"], align_model, align_metadata, input_audio, device, interpolate_method=interpolate_method, return_char_alignments=return_char_alignments, print_progress=print_progress)
            results.append((result, audio_path))

        # Unload align model
        del align_model
        gc.collect()
        torch.cuda.empty_cache()

    # >> Diarize
    if diarize:
        if hf_token is None:
            print("Warning, no --hf_token used, needs to be saved in environment variable, otherwise will throw error loading diarization model...")
        tmp_results = results
        print(">>Performing diarization...")
        results = []
        diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)
        for result, input_audio_path in tmp_results:
            diarize_segments = diarize_model(input_audio_path, min_speakers=min_speakers, max_speakers=max_speakers)
            result = assign_word_speakers(diarize_segments, result)
            results.append((result, input_audio_path))
    # >> Write
    for result, audio_path in results:
        result["language"] = align_language
        if save_results:
            writer(result, audio_path, writer_args)
        terminal_writer(result, audio_path, writer_args)


if __name__ == "__main__":
    cli()
