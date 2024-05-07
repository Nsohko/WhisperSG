import os

import textgrid
import re
from pydub import AudioSegment


class args_class():
    # path to part 3 as provided by IMDA
    part_3_path = PATH_TO_PART_3

    # path to save all output aduio files
    output_path = PATH_TO_OUTPUT

    # which mics to process. They correspond as follows:
    # 1: "Audio Same BoundaryMic",
    # 2: "Audio Same CloseMic",
    # 3: "Audio Separate IVR",
    # 4: "Audio Separate StandingMic"

    mic_ids = [1, 2, 3, 4]

    # Path to save resultant txt files for audio and script
    script_txt = os.path.join(output_path, "txt", "text.txt")
    audio_txt = os.path.join(output_path, "txt", "audio_paths.txt")


args = args_class()


# helper function to format the transcriptions, and remove any notations
# can be modified as necessary
# note that all transcriptions for part 3 are only normalized by default,
def format_string(txt):

    # _ => acronyms - Safe to remove (eg N_A_F_A -> NAFA)
    # ! => interjections (e.g Walao) - Safe to remove (!walao! -> walao). Might want to consider removing this word tho
    # (ppb) / (ppc) / (ppl) / (ppo) / <SPK/> => paralinguistic pheonomena (breath, cough, laugh, others) - Safe to remove
    # # => other languages - safe to remove (e.g #pasar malam# => pasar malam)
    # <S> => Short Pause - Safe to remove
    # <FIL/>

    remove_list = ['_', '!', '(ppb)', '(ppc)', '(ppl)', '(ppo)', '<SPK/>' '#', '<S>', '<NON/>', '<FIL/>', '<s/>',
                   '<c/>', '<q/>', '<STA/>', '<NPS/>']

    for char in remove_list:
        txt = txt.replace(char, '')
    txt = txt.replace('#', '')
    # [] => discourse particles ( eg [oh] / [ah] / [wah]) - Here we will remove the entire word from the transcript. Not sure if this is best tho (alternative is to remove only brackets but keep word)
    txt = re.sub(r'\[.*?\]\ *', '', txt)

    # () => filler particles ( eg (uh) / (um) ) - Again here we will remove the entire word
    txt = re.sub(r'\(.*?\)\ *', '', txt)

    # ~ => Incomplete words (eg abbre~ abbrev~ abbreviation) - Again here will remove the partial words
    txt = re.sub(r'(\w+~ *)', '', txt)

    # - => multi-word nouns - Replace with a space (eg Hong-Kong -> Hong Kong)
    txt = txt.replace('-', ' ')

    return txt


# helps prepare scripts for part 3
# for part 3, the same script maps to more than one audio file. Thus, we need to use repeat_id to keep track of whether we are processing the script first time second etc. This is also used as an uique id to tell us whihc mic is being used
def prepare_part3_script(script_folder, output_script_txt, mic_id):
    # a dictionary used to keep track of data of each script line
    timestamps = {}

    # open the file to be appended
    with open(output_script_txt, 'a', encoding='utf-8-sig') as outfile:

        # iterate through all combined scripts
        for combined_script in sorted(os.listdir(script_folder)):

            # get the name and path of the combined_script
            combined_script_name = combined_script[0:-9]
            combined_script_path = os.path.join(script_folder, combined_script)

            # use textgrid module
            # note that some of the scripts are formatted incorrectly, so if we come across one of them and it gives and error, we can just skip to the next one
            try:
                tg = textgrid.TextGrid.fromFile(combined_script_path)
            except:
                continue

            # unique index to keep track of each sentence within the script
            sentence_id = 0

            for data in tg[0]:
                # get the unique id of the current line
                # sentence_id corresponds to the id within the textgrid file
                # text_id is a global unique id
                sentence_id += 1
                text_id = combined_script_name + '-' + str(sentence_id) + '-' + str(mic_id)

                raw_text = data.mark
                invalid_str = False

                # if any of these flags are in the raw_text, the text is most likely not useable, so we can skip
                for flag in ['<UNK>', '<Z>', '<NEN>', '**', '<NON/>', '<S>']:
                    if flag in raw_text:
                        invalid_str = True
                        break

                # formats string using helper function defined above
                formatted_text = format_string(raw_text)

                # again if there are any of these characters, the text is inavlid, so we skip it
                for flag in ['<', '>', "/", "\\"]:
                    if flag in formatted_text:
                        invalid_str = True
                        break

                # note that part 3 onwards only has normalized scripts, sicne they did not transcribe punctuation / capitalisation / etc.

                # as mentioned, if our text was invalid, skip it
                if invalid_str:
                    continue

                print(text_id, formatted_text)
                outfile.write(text_id + "\t" + formatted_text + '\n')

                # update timestamps. It will be used later for the audio
                if combined_script_name in timestamps:
                    timestamps[combined_script_name] += [{'sentence_id': str(sentence_id),
                                                          'start': data.minTime,
                                                          'end': data.maxTime}]


                else:
                    timestamps[combined_script_name] = [{'sentence_id': str(sentence_id),
                                                         'start': data.minTime,
                                                         'end': data.maxTime}]

    return timestamps


# helper function to split audio based on start and end
def audio_split(old_address, new_address, start, end):
    # note that i give a small buffer on 10ms on either side to ensure full coverage of the audio
    start = start * 1000 - 10
    end = end * 1000 + 30

    newAudio = AudioSegment.from_wav(old_address)
    newAudio = newAudio[start:end]
    newAudio.export(new_address, format="wav")


def prepare_part3_audio(audio_folder, output_audio_txt, output_audio, mic_type, timestamps):
    # open output txt in append mode
    with open(output_audio_txt, 'a', encoding="utf-8-sig") as outfile:

        # there are 4 types of mics used in part 3, each of which are processed differently. They each have a unique mic_id which Ive assigned as follows:
        # 1: "Audio Same BoundaryMic",
        # 2: "Audio Same CloseMic",
        # 3: "Audio Separate IVR",
        # 4: "Audio Separate StandingMic"

        # 1st type
        if mic_type == "Audio Same BoundaryMic":

            mic_id = 1

            # iterate through audio files. For this mic_type, the same file provides audio for both speakers
            for audio in sorted(os.listdir(audio_folder)):
                print(audio)
                # get the audio_path and name
                audio_path = os.path.join(audio_folder, audio)
                audio_name = audio[0:-4]

                # each file represents 2 speakers, so we need to do this twice
                for speaker_id in (1, 2):

                    # this is the name of the script that thus file/speaker_id corresponds to
                    script_name = audio_name + '-' + str(speaker_id)

                    # get all the script objects corresponding to this script
                    # scripts is a list of dictionaries with the sentence_id, start time and stop time
                    # if this audio did not have a corresponding transcription, skip it
                    try:
                        scripts = timestamps[script_name]
                    except:
                        continue

                    # iterate through each of the scripts
                    for script in scripts:
                        # for the current script, the unique audio_id will be the <script_name>-<sentence_id>-<mic_id>
                        audio_id = script_name + '-' + script['sentence_id'] + '-' + str(mic_id)

                        # get the start and end times
                        start = script['start']
                        end = script['end']

                        # split audio using helper function
                        split_audio_path = os.path.join(output_audio, audio_id + ".wav")
                        audio_split(audio_path, split_audio_path, start, end)

                        # write to outfile
                        outfile.write(audio_id + "\t" + split_audio_path + "\n")

        # 2nd type
        elif mic_type == "Audio Same CloseMic":

            mic_id = 2

            for audio in sorted(os.listdir(audio_folder)):
                print(audio)

                audio_path = os.path.join(audio_folder, audio)
                script_name = audio[0:-4]

                # for this one, the audio files are already separated by speaker, so I do not need to loop throgh twice for each speaker
                try:
                    scripts = timestamps[script_name]
                except:
                    continue

                for script in scripts:
                    audio_id = script_name + '-' + script['sentence_id'] + '-' + str(mic_id)

                    start = script['start']
                    end = script['end']

                    split_audio_path = os.path.join(output_audio, audio_id + ".wav")
                    audio_split(audio_path, split_audio_path, start, end)

                    outfile.write(audio_id + "\t" + split_audio_path + "\n")

        # 3rd type
        elif mic_type == "Audio Separate IVR":

            mic_id = 3

            # for this mic, the audio folder doesnt lead directly into the audio
            # instead it leads into subfolders for eahc conversation, which in turn hold 2x audio files (one for each speaker)

            # iterate through each conversation in the folder
            for convo in sorted(os.listdir(audio_folder)):

                convo_folder = os.path.join(audio_folder, convo)

                # iterate through the 2x audio files in the conversation
                for audio in sorted(os.listdir(convo_folder)):

                    # get the audio path and corresponding script name
                    audio_path = os.path.join(convo_folder, audio)
                    script_name = convo + "_" + audio[0:-4]

                    try:
                        scripts = timestamps[script_name]
                    except:
                        continue

                    for script in scripts:
                        audio_id = script_name + '-' + script['sentence_id'] + '-' + str(mic_id)
                        print(audio_id)

                        start = script['start']
                        end = script['end']

                        split_audio_path = os.path.join(output_audio, audio_id + ".wav")
                        audio_split(audio_path, split_audio_path, start, end)

                        outfile.write(audio_id + "\t" + split_audio_path + "\n")

        elif mic_type == "Audio Separate StandingMic":

            mic_id = 4

            for audio in sorted(os.listdir(audio_folder)):
                print(audio)
                # get audio path and script name
                audio_path = os.path.join(audio_folder, audio)
                script_name = audio[0:-4]

                try:
                    scripts = timestamps[script_name]
                except:
                    continue

                for script in scripts:
                    audio_id = script_name + '-' + script['sentence_id'] + '-' + str(mic_id)

                    start = script['start']
                    end = script['end']

                    split_audio_path = os.path.join(output_audio, audio_id + ".wav")
                    audio_split(audio_path, split_audio_path, start, end)

                    outfile.write(audio_id + "\t" + split_audio_path + "\n")


def prepare_part_3(part_3_path, output_script_txt, output_audio_txt, output_audio, mic_ids):
    # arranged in ascending mic_id ( from 1 to 4 )
    all_mic_ids = {1: "Audio Same BoundaryMic",
                   2: "Audio Same CloseMic",
                   3: "Audio Separate IVR",
                   4: "Audio Separate StandingMic"}

    for mic_id in all_mic_ids:

        if mic_id not in mic_ids:
            continue

        if mic_id == 1 or mic_id == 2:
            script_folder = os.path.join(part_3_path, 'Scripts Same')
        else:
            script_folder = os.path.join(part_3_path, 'Scripts Separate')

        audio_folder = os.path.join(part_3_path, mic_ids[mic_id])

        timestamps = prepare_part3_script(script_folder, output_script_txt, mic_id)

        prepare_part3_audio(audio_folder, output_audio_txt, output_audio, mic_ids[mic_id], timestamps)


prepare_part_3(args.part_3_path, args.script_txt, args.audio_txt, args.output_path, args.mic_ids)
