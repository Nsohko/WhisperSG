from datasets import Dataset, Audio, Value
import random
import time

class args_class():

    script_txt = r"C:\Users\userAdmin\Desktop\part1_2_no_repeats_txt\text.txt"
    audio_txt = r"C:\Users\userAdmin\Desktop\part1_2_no_repeats_txt\audio_paths.txt"

    # whether to remove duplicate transcriptions (as there are multiple audios where idfferent speakers say the same sentence)
    # if set to True, randomly chooses one audio for each transcription
    remove_duplicates = True

    # where to save output .arrow files
    output_dir = r"C:\Users\userAdmin\Desktop\part1_2_no_repeat"

args = args_class()


def verify_files(script_txt, audio_txt):

    with open(audio_txt, 'r', encoding="utf-8-sig") as audio_file:

        audio_lines = audio_file.readlines()

        with open(script_txt, 'r+', encoding="utf-8-sig") as text_file:

            text_lines = text_file.readlines()

            # note this assumes there is an excess of text_lines (as this is usally what happens assuming my data laoding script is used)
            # if for some reason an excess of audio_lines is obtained instead, it should be easy enough to modify this to remove audi_lines instead
            while len(text_lines) >= len(audio_lines):

                matched = True

                for i, line in enumerate(text_lines):
                    # some of the files have an _edited_ flag in their name, we can just remove this
                    if "_edited_" in line:
                        new_line = line.replace("_edited_", "")

                        text_lines[i] = new_line
                        line = new_line

                    audio_index = audio_lines[i].split('\t')[0]
                    text_index = line.split('\t')[0]

                    if text_index != audio_index:
                        print("REMOVED")
                        print("Line Index: ", i)
                        print ("Text Line: ", line)
                        print("Audio Line: ", audio_lines[i])

                        matched = False
                        text_lines.pop(i)
                        print("Script Length: ", len(text_lines))
                        print("Audio Length: ", len(audio_lines), '\n' )
                        break

                if matched:
                    break
                
            # if we break out of the loop without ahcieving "matched", return False
            if not matched:
                print("There is an error, as the script and audio entries do not match up.")
                return False

            # otherwise we shall continue as per normal
            # we will now update the script txt files
            text_file.seek(0)
            text_file.truncate()
            text_file.writelines(text_lines)

            print("PERFORMING FINAL VERIFICATION...")
            text_lines = text_file.readlines()
            audio_lines = audio_file.readlines()

            for i, line in enumerate(text_lines):
                audio_index = audio_lines[i].split('\t')[0]
                text_index = line.split('\t')[0]

                if audio_index != text_index:
                    print("ERROR. Please revert to a backup of the script and audio txt files")
                    return False


            print( "ALL LINES MATCH" )
            return True

# eliminates entries that have the same transcription
# e.g. if two audio files are both transcribed to the same sentence, this will randomly pick one
# reduces effect of overfitting
def remove_duplicates(script_txt, audio_txt):

    with open(script_txt, 'r+', encoding='utf-8-sig') as text_file, open(audio_txt, 'r+', encoding='utf-8-sig') as audio_file:

        # first let's get data from the audiofile
        audio_lines = audio_file.readlines()
        # dictionary that stores audio_id: audio_path pairs
        id_audiopath = {}

        for line in audio_lines:
            id = line.split('\t')[0]
            audio_path = line.split('\t')[1]

            id_audiopath[id] = audio_path

        text_lines = text_file.readlines()

        # dictionary consisting of text:[ids] pairs
        text_ids = {}

        for line in text_lines:
            id = line.split('\t')[0]
            text = line.split('\t')[1]

            if text in text_ids:
                text_ids[text] += [id]
            else:
                text_ids[text] = [id]

        # unique pairs of id and their corresponding {text, audio_path} also stored in a dict
        unique_data = {}

        for text in text_ids:

            ids = text_ids[text]
            chosen_id = random.choice(ids)

            # if our chosen_id does not have a corresponding audio_file
            while (chosen_id not in id_audiopath) and len(ids) > 0:
                # remove chosen_id and choose a new id
                ids.remove(id)
                chosen_id = random.choice(ids)

            # if we removed all id from ids without finding a corresponding audio file, we can just skip this audio
            if len(ids) == 0:
                continue

            # let's get the corresponding audio_path
            audio_path = id_audiopath[chosen_id]

            # else lets write to our dictionary with our chosen_id
            unique_data[chosen_id] = {"text": text, "audio_path": audio_path}

        # now that we have gotten all our unique data lets write to our files
        text_file.seek(0)
        text_file.truncate()

        audio_file.seek(0)
        audio_file.truncate()

        print(f"Found {len(unique_data)} unique entires. Rewriting files...")
        for id in unique_data:
            text_file.write(id + "\t" + unique_data[id]["text"])
            audio_file.write(id + "\t" + unique_data[id]["audio_path"])

        print("All duplicate transcriptions removed...")

def save_data(script_txt, audio_txt):

    print("ENSURE YOU HAVE MADE BACKUPS OF YOUR SCRIPT AND AUDIO TXT FILES")
    print("THIS PROGRAM WILL DIRECTLY MODIFY THESE FILES, SO IF THERE IS AN ERROR YOU WILL NEED TO REVERT THE DOCUMENTS TO A BACKUP MANUALLY")
    print("SLEEPING FOR 10 SECONDS... ENSURE BACKUP IS CREATED, OTHERWISE TERMINATE PROGRAM NOW")

    time.sleep(10)

    print("STARTING PROCESS...")

    print("VERIFYING ENTRY ORDER...")
    if not verify_files(script_txt, audio_txt):
        return

    if args.remove_duplicates:
        print("ATTEMPTING TO REMOVE DUPLICATES")
        remove_duplicates(script_txt, audio_txt)
    
    txt_entries = open(script_txt, 'r', encoding="utf-8-sig").readlines()
    scp_entries = open(audio_txt, 'r', encoding="utf-8").readlines()

    if len(scp_entries) == len(txt_entries):
        audio_dataset = Dataset.from_dict({"audio": [audio_path.split()[1].strip() for audio_path in scp_entries],
                        "sentence": [' '.join(text_line.split()[1:]).strip() for text_line in txt_entries]})

        audio_dataset = audio_dataset.cast_column("audio", Audio(sampling_rate=16_000))
        audio_dataset = audio_dataset.cast_column("sentence", Value("string"))
        audio_dataset.save_to_disk(args.output_dir)
        print('Data preparation done')
        return True

    else:
        print('Error: Please re-check the audio_paths and text files. They seem to have a mismatch in terms of the number of entries. Both these files should be carrying the same number of lines.')
        return False

save_data(args.script_txt, args.audio_txt)

