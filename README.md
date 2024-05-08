# whisperSG
conda create --name whispersg python=3.9
conda activate whispersg
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install ffmpeg=4.2.2
git clone https://github.com/Nsohko/whisperSG
cd whispersg
pip install -e .
huggingface-cli login
## 0. Background

This project aims to explore the creation of an Automatic Speech Recognition (ASR) tool to accurately transcribe Singaporean speakers in a variety of contexts. This will ideally also involve an ability to recognise Singlish, as well as other uniquely Singaporean / Southeast-asian terms (e.g _Nasi Lemak_ ).

The project methodology involved fine-tuning a pre-existing ASR model using the National Speech Corpus provided by IMDA. We then also tested potential avenues of deploying the software, and its accuracy in transcribing both pre-recorded and real-time data. The specific model we chose was Whisper, which was developed by OpenAI in 2022. More details on the training data and model architecture are below.

Through this project, we also explored how we could integrate our finetuned Whisper with other models to provide even more functionality. Some examples include a speech diarization model to help differentiate and identify speakers, and a Large Language Model to summarise transcribed text.

Finally, we also sought to to streamline this whole finetuning/deployment/integration pathway to better enable future teams to more effectively create their own local ASR solutions using custom data.

### 0.1. Applications

As one of our main goals was to catalogue and streamline the process of creating a custom ASR software, we believe the primary value of this project will be in enabling future teams to more effectively create local ASR solutions that can meet their specific needs.

However, besides that, the example carried out in this project (finetuning Whisper using Singaporean speech) can also have numerous applications. Some examples are as follows:

1. To transcribe and store voice communications during critical periods, which can provide an additional dimension of analysis

2. To automatically transcribe meetings for future reference

3. In the case of accessibility, to allow those who may have difficulty typing / interacting with devices to use voice-commands instead

### 0.3. Usage

Please follows the steps below for the detailed instructions on how to set up the environment.

The usage of this project is largely imported from WhisperX, which is a command-line app

For direct usage on an audio file:

To record data with microphone instead:

To try live trascription (Prototype): 

## 1. Project Information

#### 1.1. WhisperX

As mentioned previously, the model we have chosen for this project is Whisper. However, in particular, we have chosen to use the WhisperX version, developed by Max Bain (m-bain) [here](https://github.com/m-bain/whisperX?tab=BSD-4-Clause-1-ov-file).

The base whisper is a multitasking general purpose ASR model. It can perform multilingual speech recognition, speech translation, speech recognition and language identification. However, in this project, we will be focusing on its use in performing automatic speech recognition in English.

Whisper uses a transformers-based sequence-to-sequence architecture. It consists of two "sub-models" in the form of an encoder and decoder. Essentially, whisper takes in a sequence of audio in the form of a log-Mel spectogram which is then then encoded by the encoder into a sequence of hidden states. The decoder then decodes these hidden states to predict text tokens.  An attention mechanism is also used to refine the output based on previous tokens. A diagram is as follows:

![image](https://github.com/Nsohko/SGWhisper/assets/102672238/6b776e8e-a47c-4954-8628-43d3e4bbcdfc)

Whisper comes in 5 sizes, as listed below. The tiny, base, small and medium versions also come in English-only versions (i.e trained only on English data). For the purposes, of this project we will be focusing on the tiny and small models

![image](https://github.com/Nsohko/SGWhisper/assets/102672238/0ea5d171-6590-4987-a566-8fec67f834ce)

WhisperX is a refinement to whisper, developed by Max Bain. It use the [faster-whisper](https://github.com/SYSTRAN/faster-whisper) backend to achieve much faster trasncription with lower GPU memory requirements. It also provides integration with other models to achive greater functionality, including forced-alignment, world-level timestamps and speaker diarization. In particular, the speaker diarization is provided by models from [pyannote-audio](https://github.com/pyannote/pyannote-audio)

The faster-whisper backend makes use of [CTranslate2](https://github.com/OpenNMT/CTranslate2), which is fast inference engine for transformer models, to greatly increase the efficiency of whisper.

### 1.2. Training Data

The training data we chose was the National Speech Corpus, provided by IMDA Singapore.

However, we can finetune our model with nearly any source of audio. The dataset preparation process will be elaborated on later.

#### 1.2.1. IMDA Dataset

The National Speech Corpus is an open-source English Corpus provided by the Info-communications and Media Development Authority (IMDA) of Singapore. It consists of audio recording and transcriptions of Singaporean speakers in a variety of contexts. This dataste is extremely large and it has a total size of ~1.8TB

All the audio recording in the dataset are saved as .wav files. The transcriptions are either saved as .txt files (Part 1/2) or .TextGrid files (Part 3 onwards)

The data is split up into 6 parts, however we will only be using part 2 for this project. This provides us with more than sufficient training data to achieve a relatively significant imporvement of the model's accuracy after finetuning, while also being manageable within the scope of this project. However, as an extension, I have provided the code to pre-process parts 1 and 3 as well, so it should be relatively easy to expand the training data for projects with a larger scope.

A bried description of the first 3 parts of the IMDA dataset are as follow:

##### 1.2.1.1. Part 1 / Part 2

Both part 1 and part have largely the same format of data. The primary difference is that part 1 consists on phoenetically balanced scripts, while part 2 focuses more on sentences that that use uniquely Singaporean words and names.

The Tree structure for the data is summarises roughly as follows:

![image](https://github.com/Nsohko/SGWhisper/assets/102672238/30b247f6-5af7-40b4-afab-08c42743eda8)


Channel 0 / 1 / 2 all contain the same recordings organized in the same manner, just recorded with different devices :-

Channel 0: Headset / Professional Standing Microphone

Channel 1: Boundary Microphone (placed far from speaker)

Channel 2: Mobile Phone

IMDA has also provided metatdata on the sex and race of the speakers.

For this project, I have chosen to finetune Whisper solely on Channel 0 of part 2. This would give us over 820, 000 unique high-quality audio recordings and their corresponding transcriptions. This is more than enough to achieve a relatively significant improvement in whisper's accuracy, while also being manageable within the scope of our project. I chose part 2 in particular since it consists of sentences with an emphasis on uniquely Singaporean terms, which I felt would be most useful and applicable for our model to be finetuned on.

##### 1.2.1.2. Part 3

Part 3 consists of conversational data between two speakers. It is divided into conversations where the speakers are in the same room, and when they are in seprate  rooms. In both cases, the transcriptions are done on a 'per-speaker' basis, i.e for a particular conversation, there are two transcription documents corresponding to each individual speaker. The transcriptions are stored in a Praat-Textgrid format

For the set of recording where the speakers are in the same room, the conversations were recorded using both a Boundary Mic and a Close Mic. The Boundary Mic is able to pick-up the audio form both speakers simultaneosuly, so each conversation corresponds to one Boundary Mic Recording. On the other hand the Close Mic is only able to pick up audio from its own speaker, so each conversation has two corresponding Close Mic recordings (one for each speaker)

Similarly, for the set of the rcordings where the speakers are in different room, the conversations were recorded using both Standing Mics and IVR (Telephone). These both onlu pick up audio from their own corresponding speaker, resulting in 2 audio recording per conversation (corresponding to each speaker) for these microphones.

To summarise:

For a conversation where the speakers are in the same room, there will be:
1. 2x Transcriptions (for each speaker)
2. 1x Recording using Boundary Mic (records both speakers simultaneosuly)
3. 2x Recording using Close Mic (for each speaker)

For a conversation where the speakers are in different rooms, there will be:
1. 2x Transcriptions (for each speaker)
2. 2x Recording using Standing Mic (for each speaker)
3. 2x Recording using IVR (for each speaker)

Part 3 also contains a "DOCUMENTS" subfolder which contain speaker metadata (similar to Part 1 / 2), but I have not included it since we will not be using it in this project

Here is a rough digram of the tree structure for Part 3
 
![image](https://github.com/Nsohko/SGWhisper/assets/102672238/32812c4f-b247-4adf-8bd5-f86df2189bfc)


#### 1.2.2. Dataset Preparation

We will next need to pre-process the dataset before it can be used for finetuning with the huggingface's sequence-to-sequence training pipeline. In particular, we will need the database to be stored in the .arrow chunks.

We will leverage the scripts provided by the user **vasistalodagala** [here](https://github.com/vasistalodagala/whisper-finetune)) to do this.

The slightly modified version of his script used in this project expects two txt files, named audio_paths.txt and text.txt

The audio_paths.txt file consists of pairs of unique audio IDs and the absolute file path to the corresponding audio file. It should be formatted as such:

```
<unique-id> <absolute path to the audio file-1>
<unique-id> <absolute path to the audio file-2>
...
<unique-id> <absolute path to the audio file-N>
```

The text.txt file should contain the transcriptions corresponding to each of the audio files mentioned in the audio_paths file. They should also be indexed by the corresponding audio files's unique Audio ID (as stored in audio_paths.txt). 

Note: The length of text.txt and audio_paths.txt should be the exact same, and the ordering of audio IDs in both files should be consistent as well

The text.txt file should be formatted as such:

```
<unique-id> <Transcription (ground truth) corresponding to the audio file-1>
<unique-id> <Transcription (ground truth) corresponding to the audio file-2>
...
<unique-id> <Transcription (ground truth) corresponding to the audio file-N>
```

I have written a script to directly convert parts 1-3 of the raw IMDA dataset into a local copy stored in the format above. After which, vasistalodagala's script can be used to convert the data into final format required for whisper. I have opted to leave these 2 scripts separate, so that if you want to use a different dataset other than the IMDA NSC, you can just access vasistalodagala's script directly. However, you would be required to complete the intermediate step of independently generating the two text files as per above first.

If using parts 1-3 of the IMDA dataset, my script can handle all of the pre-processing required. This includes both organizing all the audio files, as well as ensuring all the transcriptions are valid and accurate by formatting the text and removing any unnecessary notations. For part 3, the script also slices the audio files based on the timestamps provided in the transcription document. This ensures that each segment of transcription can be mapped to just one audio file.

As a proof-of-concept for this project, I opted to use only Channel 0 of part 2 to finetune Whisper, which consists of over 820, 000 unique high-quality audio-transcription pairs. This is significant enough to measure a decently large improvemnt in whisper's accuracy, while also being managebale within the short timeframe and limited hardware capacity of this project.

Where possible, I tried to diversify the quality and source of the audio data to hopefully create a model that is more resilient to unpredictable data.

### 1.3. Fine-tuning Whisper

The fine-tuning process has been largely abstracted away by the Huggingface Datasets and Transformers. As such, once we pre-process the data, the fine-tuning process should be rather smooth.

As far as possible, I have tried to optimise to reduce disc space consumption for the process to better allow for large datasets (inlcuding the IMDA NSC) to be used more effectively. However, this does come at the cost of some time efficiency.

During the finetuning process, I also opted to normalize all the training data (i.e. make all transcriptions lower-case and remove punctuations) as I wanted to focus solely on teaching the model to recognise the Singaporean context and vocabulary, without a focus on casing and punctuation. However, this can be easily modified if necessary.

### 1.4. Speech Diarization

The integration between my finetuned whisper model and a speaker diarization model is handled by the WhisperX backend.

However, WhisperX requires a faster-whisper model, which is a reimplementation of the base Whisper model using CTranslate2. Thus, to create compatibility, we will first need to convert our model. Luckily, this can be easily done using the Python CTranslate2 library which already provides in-built support for models hosted on HuggingFace transformers, including whisper.

WhisperX's multispeaker diarization is provided by the pyannote-audio library.

elabprate on pyannote

### 1.5. Text Summarization

As an extension to this project, I also explored the integration of a Large Language Model to aid in summarising the transcribed text.

The model I ultimately chose is BART (Bidirectional and Auto-Regressive Transformer). Similar to Whisper, this also uses a transformer-based architecture, consisting of encoders and decoders.

I chose the model in particular due to its relatively decent performance out-of-the-box, and its large amount of available documentation online (given that it is open-source). Moreover, compared to other models like LLaMA 2, BART is relatively lightweight and does not need as significant computational resources. 

However, one of the largest limitations of BART is its relatively small input token size limit (1024). This makes it much less effective at summarising long chunks of data that exceed this limit. To address this, I implemented a recursive algorithm that chunks long text into smaller pieces within the limit and feeds it into the model one at a time. The summarised chunks are then concatenated, and the process is repeated till the summary fits within the model's token limit. While this strategy does work, it tends to take an exponential amount of time, and summary quality does tend to degrade on longer tasks.

In this future, I hope to increase the quality of the summaries by either choosing newer and more mdoern models with larger token limits, or fine-tuning the pre-trained bart to handle much longer inputs

### 1.5. Final Application

The usage of this is largely inherited from WhisperX, which is a command-line app. The main command arguments are:

I have also integrated somewhat real-time transcription support, which can be accessed by:

This is still largely a prototype, as it far from working perfectly (especially on devices with lower compute power). The difficulty in achieving real-time transcription with whisper is lagely due to the nature of the model itself, as it was built to transcribe audio in chunks of 30 seconds. As such, it is not able to transcribe audio 'on-the-fly'. This problem is even more true for whisperx, whose diarization models were also similarly trained to diarize entire audio files at a time, rather than in rea-time.

However, there are some workarounds to this, which I've implemented above.

1. 

## 2. Step-by-step Guide to setting up

Before getting started, please install anaconda/miniconda onto your device. If using the IMDA dataset, please also install dropbox and s request access to the datset [here](https://www.imda.gov.sg/how-we-can-help/national-speech-corpus)

Please also ensure that your device has at least 64GB of RAM, 1.5TB of storage space and a CUDA 12.X compatibe NVIDIA GPU.

### 2.1 Setting up the environment
`conda create --name whisperSG python=3.8`

`conda activate whisperSG`

Next we will install torch. The commands to install different versions of torch (along with their corresponding CUDA compilations) can be found [here](https://pytorch.org/get-started/locally/). For the purposes of this project, it has been tested using PyTorch 2.2.2 using CUDA 12.1/

You can install it with the following command:
`conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`

Next install ffmpeg < 4.4 as follows: 
`conda install -c conda-forge ffmpeg<4.4`

Clone this repo:
`https://github.com/Nsohko/whisperSG.git`

Navigate to the repo:
`cd whisperSG`

1. how can i use it? pip install -> cmd line?
   pip install -e .
3. ensuring default model is correct
4. record and execute
5. starting live
6. gradio app
7. navigate to trianing, and execute training


## 2. Review

### 2.1. Results

After finetuning our model on a subset of the IMDA National Speech Corpus for 2 epochs, we were able to achieve the following improvements:

With our best word error rate (wer) being ~4%, this is easily comparable to other State-of-The-Art (SoTA) speech recognition models in other areas.

As such, it does seem like the finetuning on the IMDA NSC is a viable strategy to develop a strong and accurate speech recognition model for Singaporean Speaker. Moreover, this was achieved using a relatively small ( < ~1/6) subset of the entire corpus, meaning we could potentially improve performance further. Moreover, we only finetuned the model with a single quality of audio recordings. By instead using the much larger variety and qualities of recordings provided in the rest of the IMDA corpus, we could also build a model that is more resilient in a wider range of environments

### 2.2 Limitations

### 2.3 Extensions

.

### 2.2 Reflection

Overall, this has been an extremely exciting and enjoyable project. As someone who had extremely limited computer science and coding knowledge prior, I also really appreciate that this project was able to provide me with such a strong introduction to the field. 


