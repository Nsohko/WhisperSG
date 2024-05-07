import os
import shutil
from zipfile import ZipFile


class args_class():
    # whether to use normalized scripts for the training data
    # normalized_script = False

    # path to either part 1 or part 2 (in the same format as used by IMDA)
    part_path = r"C:\Users\userAdmin\Dropbox\IMDA - National Speech Corpus\PART1"

    # set channels to a list of which channels to process
    channels = ["CHANNEL2"]

    # whether to use normalized_script. Recommended to be False so that whisper can be finetuned on punctuation as well
    normalized_script = False

    # path to save extracted audio files
    output_path = r"C:\Users\userAdmin\Desktop\part1_2"

    # path to save final txt files for script and audio.
    script_txt = os.path.join(output_path, "txt", "text.txt")
    audio_txt = os.path.join(output_path, "txt", "audio_paths.txt")


args = args_class()


# This is a helper function to process a single channel for Part 1 / Part 2

# script_folder the path of a folder containing all scripts for the channel
# script_txt is the path to txt file where we will put our formatted script names

# audio_folder is the folder with the audio for the current channel
# audio_txt is the path to the txt file where we wil store the audio paths
# output_path is the folder where we will store our formatted data

def prepare_channel(script_folder, audio_folder, script_txt, audio_txt, output_path):
    seen = set()

    # open the output_script txt in append mode. the reason for append is so that we can combine diff channels later
    with open(script_txt, 'a', encoding="utf-8-sig") as outfile:

        # iterate through each of the scripts in the channel
        for script in sorted(os.listdir(script_folder)):

            # get the script path
            script_path = os.path.join(script_folder, script)

            # open the script in read
            with open(script_path, 'r', encoding="utf-8-sig") as infile:

                # iterate through each line in the script
                for i, line in enumerate(infile):

                    # if we dont want the normalized script, use the method below
                    if not args.normalized_script:

                        if i % 2 == 0:
                            print(line)
                            key = line.split("\t")[0]
                            text = line.split("\t")[1]

                            # check if there are any invalid chars, if so skip this one
                            invalid_chars = ['<', '>', '/', '\\']
                            for invalid_char in invalid_chars:
                                if invalid_char in text:
                                    continue

                            seen.add(key)

                            outfile.write(line)
                        else:
                            continue

                    # otherwise get the normalized text
                    else:
                        if i % 2 == 0:
                            key = line.split("\t")[0]
                            print(key, end="\t")

                            seen.add(key)

                            outfile.write(key + "\t")
                        else:
                            transcript = line.strip()

                            print(transcript)

                            outfile.write(transcript + "\n")

    ### Done with script prep, moving onto audio prep ###

    # temp folder to store extracted audio files
    temp = output_path + r'\temp'

    # clear and make temp
    if os.path.exists(temp):
        shutil.rmtree(temp)
    os.makedirs(temp)

    # iterate through the zip files with the audio
    for speaker_zip in sorted(os.listdir(audio_folder)):
        print(speaker_zip)
        # the address to the current zip file
        speaker_zip_path = os.path.join(audio_folder, speaker_zip)

        # extract all the audio files to temp
        with ZipFile(speaker_zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp)

        # the zip extracts itself into a folder, so we need to drill into that folder first
        for person in sorted(os.listdir(temp)):

            person_path = os.path.join(temp, person)

            # iterate through each session
            for session in sorted(os.listdir(person_path)):

                # get the path to the session
                session_path = os.path.join(person_path, session)

                # iterate through the recordings in each channel
                for recording in sorted(os.listdir(session_path)):
                    # move the extracted audio file into the actual directory
                    recording_name = recording[0:-4]

                    # if we haven't seen its corresponding transcription, skip it
                    if recording_name not in seen:
                        continue

                    current_recording_path = os.path.join(session_path, recording)
                    new_recording_path = os.path.join(output_path, recording)

                    os.rename(current_recording_path, new_recording_path)

            # remove the extracted folder from temp
            shutil.rmtree(person_path)

    # remove the temp directory
    shutil.rmtree(temp)

    # open the txt file in append mode
    with open(audio_txt, 'w') as outfile:

        # iterate through all the extracted audios
        for audio in sorted(os.listdir(output_path)):

            # if it isnt a .wav file, skip it
            if audio[-4:] != ".WAV":
                continue

            # get the path to audio
            audio_path = os.path.join(output_path, audio)
            audio_name = audio[0: -4]

            # write the audio_name and path
            outfile.write(audio_name + "\t" + audio_path + "\n")


# processes all channels in part 1 / part 2. requires the address to the raw part1 / part 2 downloaded directly from IMDA
def prepare_part_1or2(part_path, script_txt, audio_txt, output_path, channels):
    data_path = os.path.join(part_path, "DATA")

    for channel in sorted(os.listdir(data_path)):

        if channel not in channels:
            continue

        channel_path = os.path.join(data_path, channel)

        script_folder = os.path.join(channel_path, 'SCRIPT')
        audio_folder = os.path.join(channel_path, 'WAVE')

        prepare_channel(script_folder, audio_folder, script_txt, audio_txt, output_path)


prepare_part_1or2(args.part_path, args.script_txt, args.audio_txt, args.output_path, args.channels)
