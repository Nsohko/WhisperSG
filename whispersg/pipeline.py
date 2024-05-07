import os
import warnings

import numpy as np
import torch.cuda

from .asr import load_model, DEFAULT_ASR_MODEL_PATH
from .diarize import DiarizationPipeline,assign_word_speakers
from .alignment import load_align_model, align
from .audio import load_audio
from .utils import (LANGUAGES, TO_LANGUAGE_CODE, get_writer, optional_float,
                    optional_int, str2bool)



class WhisperSGPipeline:

    def __init__(self,
                 whisper_model_name=DEFAULT_ASR_MODEL_PATH, cache_dir=None,
                 device=None, device_index=0, batch_size=8, compute_type="float16", threads=0,

                 alignment=True, diarization=False,
                 task="transcribe", language=None,


                 align_model_name=None, interpolate_method="nearest", return_char_alignments=False,

                 vad_onset=0.400, vad_offset=0.300, chunk_size=30,

                 min_speakers=None, max_speakers=None,

                 temperature_increment_on_fallback=0.2, temperature=0,
                 beam_size=5, patience=1.0, length_penalty=1.0, best_of=5,

                 suppress_tokens=[-1], suppress_numerals=False,

                 initial_prompt=None, condition_on_previous_text=False,

                 compression_ratio_threshold=2.4, logprob_threshold=1.0, no_speech_threshold=0.6,

                 hf_token=None):

                self.hf_token = hf_token

                self.whisper_model_name = whisper_model_name
                self.cache_dir = None

                if device is None:
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                else:
                    self.device = device
                self.device_index = device_index
                self.batch_size = batch_size
                self.compute_type = compute_type
                self.chunk_size = chunk_size

                self.faster_whisper_threads = 4
                if threads > 0:
                    torch.set_num_threads(threads)
                    self.faster_whisper_threads = threads

                self.alignment = alignment
                self.diarization = diarization

                self.task = task
                if self.task != "transcription":
                    self.alignment = False
                if task == "translate" and self.whisper_model_name.endswith('.en'):
                    raise ValueError(
                        "Translation can only be done with multilingual models (small, medium, etc.), not english-only models (e.g. small.en); Please pass --model small instead for instance")

                self.language = language
                if language is not None:
                    language = language.lower()
                    if language not in LANGUAGES:
                        if language in TO_LANGUAGE_CODE:
                            self.language = TO_LANGUAGE_CODE[language]
                        else:
                            raise ValueError(f"Unsupported language: {language}")

                if self.whisper_model_name.endswith(".en") and self.language != "en" :
                    warnings.warn(f"{self.whisper_model_name} is an English-only model; Forcing English use.")
                    self.language = "en"
                self.align_language = self.language if self.language is not None else "en"  # default to loading english if not specified

                self.align_model_name = align_model_name
                self.interpolate_method = interpolate_method
                self.return_char_alignments = return_char_alignments

                self.min_speakers = min_speakers
                self.max_speakers = max_speakers

                increment = temperature_increment_on_fallback
                if increment is not None:
                    temperatures = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
                else:
                    temperatures = [temperature]

                self.asr_options = {
                    "beam_size": beam_size,
                    "best_of": best_of,
                    "patience": patience,
                    "length_penalty": length_penalty,
                    "temperatures": temperatures,
                    "compression_ratio_threshold": compression_ratio_threshold,
                    "log_prob_threshold": logprob_threshold,
                    "no_speech_threshold": no_speech_threshold,
                    "condition_on_previous_text": condition_on_previous_text,
                    "initial_prompt": initial_prompt,
                    "suppress_tokens": suppress_tokens,
                    "suppress_numerals": suppress_numerals,
                }

                self.vad_options={"vad_onset": vad_onset,
                                  "vad_offset": vad_offset}


                # load whisper model
                self.whisper_model = load_model(self.whisper_model_name, device=self.device, device_index=self.device_index, download_root=self.cache_dir, compute_type=self.compute_type, language=self.language, asr_options=self.asr_options, vad_options=self.vad_options, task=self.task, threads=self.faster_whisper_threads)

                if self.alignment:
                    # load alignment model
                    self.align_model, self.align_metadata = load_align_model(self.language, self.device, model_name=self.align_model_name)

                if self.diarization:
                    self.diarize_model = DiarizationPipeline(use_auth_token=self.hf_token, device=self.device)

                self.timestamp_writer = get_writer('vtt', None)
                self.writer = get_writer('txt', None)
                self.writer_args = {"highlight_words": False, "max_line_count": None, "max_line_width": None}

    def get_audio_data(self, audio):
        if isinstance(audio, np.ndarray):
            audio_data = audio
        elif os.path.exists(audio):
            audio_data = load_audio(audio, 16000)
        else:
            audio_data = None

        return audio_data
    def transcribe(self, audio):

        audio_data = self.get_audio_data(audio)
        if audio_data is None:
            raise TypeError("Invalid input audio type. Please pass in either a numpy.ndarray or a string representing the path to a . wav file")


        transcription = self.whisper_model.transcribe(audio_data, batch_size=self.batch_size, chunk_size=self.chunk_size)

        if len(transcription["segments"]) == 0:
            return []

        if self.alignment:
            transcription = self.do_alignment(transcription, audio)

        if self.diarization:
            transcription = self.do_diarization(transcription, audio)

        return transcription

    def do_alignment(self, transcription, audio):
        if transcription.get("language", "en") != self.align_metadata["language"]:
            # load new language
            print(f"New language found ({transcription['language']})! Previous was ({self.align_metadata['language']}), loading new alignment model for new language...")
            self.align_model, self.align_metadata = load_align_model(transcription["language"], self.device)
            self.language = transcription['language']

        aligned_transcription = align(transcription["segments"], self.align_model, self.align_metadata, audio, self.device,
                                      interpolate_method=self.interpolate_method, return_char_alignments=self.return_char_alignments)

        return aligned_transcription

    def do_diarization(self, transcription, audio):

        diarize_segments = self.diarize_model(audio, min_speakers=self.min_speakers, max_speakers=self.max_speakers)
        diarized_transcription = assign_word_speakers(diarize_segments, transcription)

        return diarized_transcription

    def print_transcription(self, transcription, show_timestamps=True):
        if show_timestamps:
            self.timestamp_writer(transcription, None, self.writer_args)
        else:
            self.writer(transcription, None, self.writer_args)



