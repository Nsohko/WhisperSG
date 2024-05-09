# WhisperSG

## 1. Introduction

Automatic Speech Recognition (ASR) is one of the key applications of Artificial Intelligence in the field of Natural Language Processing. It is usually used in the context of automatically transcribing raw audio input into its corresponding text representation.

While LLMs have taken the spotlight in recent years, many development efforts have still been spent on building cutting-edge ASR models. In fact, just in 2022, OpenAI released its own ‘State-of-The-Art’ (SoTA) ASR model, called Whisper. Having been pre-trained on a significantly larger dataset than many of its competitors, it can perform extremely well in a variety of use cases. Moreover, beyond transcription, it can also perform additional tasks like translation and language identification.

In this project, we aim to explore the creation of an Automatic Speech Recognition (ASR) tool to accurately transcribe Singaporean speakers in a variety of contexts. To suit our specific needs and use cases, we needed to achieve two main requirements:

1.	Due to security concerns in the public sector, we require a local solution that can function on limited hardware without needing to access the internet / cloud

2.	The model must also be trained to recognize Singaporean speakers and Singlish, to improve accuracy in day-to-day use.
   
To achieve this, we will be finetuning the pre-trained Whisper model using publicly available data on Singaporean speakers.

To further enhance our product, we also explored how we could integrate it with other models to provide even greater functionality. Some examples include an alignment model to provide accurate timestamps (down to the word-level), as well as a speaker diarization model to identify and differentiate between speakers. Furthermore, we also tested how this package could be deployed and utilized for a variety of use-cases, including both asynchronous and real-time transcription, as well as other tasks like translation.

## 2. WhisperSG

### 2.1. Overview

To achieve our goals, we have explored the development of our own localized Automatic Speech Recognition tool for transcript generation.

We also finetuned our model on Singaporean speakers to help achieve greater accuracy in day-to-day use.

</br>
<p align="center">
  <img src=https://github.com/Nsohko/WhisperSG/assets/102672238/5fad8e7d-3c78-43cd-90c7-4616407ef623/>
</p>
<p align="center"><em>Overview of our solution's architecturee</em></p>
<br/>

### 2.2. Whisper

The foundational ASR model we chose for this project is [Whisper](https://github.com/openai/whisper), which is an open-source model released by OpenAI back in 2022. The foundational version from OpenAI already comes pre-trained on a whopping 680,000 hours of audio, which is significantly more that nearly any of its competitors.

Whisper uses a transformers-based architecture, very similar to those of today’s LLMs like GPT / LLaMa. This architecture excels in performing sequence-to-sequence tasks (transforming an audio sequence into a sequence of text tokens in Whisper’s case).

At a basic level, the overarching Whisper model consist of two “sub-models” called an encoder and decoder respectively. The encoder takes in raw audio input in the form of a log-Mel spectrogram and encodes it into a sequence of hidden states. The decoder then decodes these hidden states to predict the corresponding text tokens. Similar to modern LLMs, an attention mechanism is also used to simulate an understanding of long-range interactions between text tokens.

</br>
<p align="center">
  <img src=https://github.com/Nsohko/WhisperSG/assets/102672238/9c242501-e7d4-4c00-a396-2e219ee032bf />
</p>
<p align="center"><em>Whisper's transformer architecture</em></p>
<br/>

We chose Whisper over the other commonly available ASR models as it has been shown to [outperform](https://cobusgreyling.medium.com/how-will-openai-whisper-impact-current-commercial-asr-solutions-e6c683ac5940) many of its competitors (both commercial and open-source) in a variety of use cases. 

</br>
<p align="center">
  <img src=https://github.com/Nsohko/WhisperSG/assets/102672238/ec864418-21bf-4396-8ff6-0a5a24f68050 />
</p>
<p align="center"><em>
Comparison of Whisper against its competitors </br>
The y axis represents Word Error Rate (WER), with a lower value indicating a higher accuracy </br>
Column E (computer-assisted) refers to ASR followed by human verification 
</em></p>
</br>

Whisper is also relatively lightweight and comes in a variety of sizes (tiny, base, small, medium and large). The tiny, base, small and medium sizes also come in English-only (designated by a .en at the end of the name, e.g., small.en) and multilingual versions, while large only comes in a multilingual version.

For this project, we chose to finetune the small English-only version of Whisper, as we found it to have a good balance of accuracy and performance even on limited hardware. This made it manageable within the scope of our project (especially in comparison to larger models like medium/large), while also being relatively accurate when used to transcribe audio (compared to smaller models like tiny/base). The comparison between model sizes is summarized [below](https://cdn.openai.com/papers/whisper.pdf).

</br>
<p align="center">
  <img src=https://github.com/Nsohko/WhisperSG/assets/102672238/d10a9ce3-f26e-42fb-9be0-137350c41115/>
</p>
<p align="center"><em>
Comparison between different Whisper sizes </br>
<sup>1</sup>Relative Speed is as measured by OpenAI</br>
<sup>2</sup>Inference speed is as measured by us when transcribing 140 minutes of audio on an RTX3080, with WhisperX optimisations (CTranslate2 + VAD) 
</em></p>
</br>

### 2.3. Finetuning
To further improve the model’s accuracy, especially when transcribing Singaporean speakers and Singlish, we further finetuned the foundational Whisper model further using custom data. I will go into the details of our datasets in subsequent sections

</br>
<p align="center">
  <img src=https://github.com/Nsohko/WhisperSG/assets/102672238/4b9ccfb0-a401-41be-b0ec-8bd857103e7c>
</p>
<p align="center"><em>
Overview of the finetuning process
</em></p>
</br>

The raw datasets consisted of audio files (stored in the .wav format) and their corresponding transcriptions (stored as .txt files). Before we could start the finetuning process, we first had to preprocess the data into a format usable by model. This involved converting the audio files into Log-Mel Spectrograms, and formatting/cleaning any transcriptions as necessary. The datasets were also split into 2 subsets with 80% and 20% of the examples respectively. The larger subset was used for training, while the smaller one was used for an unseen evaluation of the model at the end of the finetuning.

For the finetuning itself, we used the [Huggingface transformers](https://github.com/huggingface/transformers) sequence-to-sequence trainer, which helped streamline the entire process. In total, for our final dataset of ~800,000 unique examples, the entire finetuning process (excluding data downloading/pre-processing) took about ~105 hours for 1 epoch (75hrs training + 30 hours evaluation). 

</br>
<p align="center">
  <img src=https://github.com/Nsohko/WhisperSG/assets/102672238/2f2597fa-ce83-4a20-a77a-284ec1ebd5c6>
</p>
<p align="center"><em>
Hyperparameters used for finetuning
</em></p>
</br>

### 2.4. IMDA National Speech Corpus
Our primary source of Singaporean speaker data for finetuning was the [National Speech Corpus](https://www.imda.gov.sg/how-we-can-help/national-speech-corpus), which is an open-source English Corpus provided by the Info-communications and Media Development Authority (IMDA) of Singapore. It consists of audio recordings and transcriptions of Singaporean speakers speaking both formal English and Singlish in a variety of contexts. The dataset is split into 6 parts, with part 1 focusing on formal English, part 2 focusing on Singlish and parts 3 onwards focusing on conversational recordings. All audio recordings are stored as .wav files, while the transcriptions are either saved as .txt files (parts 1 & 2) or .TextGrid files (part 3 onwards

</br>
<p align="center">
  <img src=https://github.com/Nsohko/WhisperSG/assets/102672238/27b6af57-92d6-4344-ad20-86fd2a6e417e>
</p>
<p align="center"><em>
Overview of the IMDA NSC
</em></p>
</br>

For the purposes of our project, we will be using only parts 1 and 2. This is because parts 1 and 2 have non-normalised transcription (i.e., transcriptions with punctuations and capitalisation present), while part 3 onwards only has normalised transcriptions (i.e., all punctuations and capitalisation stripped). While normalised transcriptions can be used to finetune the model, this would cause the resultant model’s predicted output to be similarly normalised (as it would ‘learn’ this normalised behaviour from the dataset). This would limit the usability of the model, as it would be very hard for human users to read its output due to a lack of punctuation and capitalisation. As such, we found it much better to only finetune the model using non-normalised transcriptions to retain the model’s ability to properly punctuate and capitalise text.

For each transcription in parts 1 and 2, there are 3 corresponding audio files, stored in different channel folders. All 3 of these audio files consist of the same recording, simply recorded with different devices as follows:

•	Channel 0: Headset / Professional Standing Microphone  
•	Channel 1: Boundary Microphone (place far from speaker)  
•	Channel 2: Mobile Phone  

For part 1, I used the recordings from Channel 2 (Mobile Phone). This was to simulate real world phone calls, and train the model to accurately transcribe such conversations over the phone. 

As for part 2, I used the recordings from Channel 0 (Headset / Professional Standing Microphone). Since part 2 was focused mainly on Singlish vocabulary, I wanted the model to ‘focus’ fully on learning these new words and adding them to its own vocabulary, hence I chose the channel with cleanest and highest audio quality. 

Overall, by combining channel 2 of part 1 and channel 0 of part 2, this helped improve the diversity of our audio recordings in terms of audio quality, which would hopefully make it suitable for effective finetuning.

### 2.5. Common Voice Dataset

One key issue with the IMDA NSC was its high degree of repetition, as there were certain words that appeared in an excessive number of transcriptions. For example, the word ‘Greenfield’ was used in over 130 of the transcriptions! During our testing, we found that this made the finetuned model extremely prone to overfitting.

To address this, we chose to augment our dataset with the [Mozilla Common Voice 17 Dataset](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) . We randomly sampled ~400,000 additional unique examples from the Common Voice dataset to dilute our original dataset and reduce the extent of repetition. We chose the Common Voice in particular due to its high diversity of transcription context and audio quality.

### 2.6. WhisperX
To further enhance the functionality of our model, we also integrated it with WhisperX . This is an interface / wrapper for Whisper that helps pre-process incoming audio and the outgoing transcriptions during inference. It was developed by Max Bain (m-bain) [here](https://github.com/m-bain/whisperX).

WhisperX has 3 main functions:
1.	Voice Activity Detection  
•	WhisperX cuts incoming raw audio based on periods of silence, and only sends the chunks with people speaking to be transcribed by the Whisper model.  
•	This helps reduce transcription time by reducing the total length of audio needed to be transcribed by Whisper (i.e., we avoid wasting time by making the Whisper model transcribe periods of silence).  

2.	Timestamp alignment  
•	WhisperX includes a phenome alignment model that assigns timestamps to specific phenomes in the incoming audio.  
•	This is then aligned with transcription output of the Whisper model to provide highly accurate timestamps for the transcription, down to the word level.  

3.	Speaker Diarization  
•	WhisperX also includes a speaker diarization model that can identify speakers and assign speaker tags to various sections of the input audio.  
•	This is again integrated with the Whisper model to differentiate speakers and add speaker identities to the transcription.  

</br>
<p align="center">
  <img src=https://github.com/Nsohko/WhisperSG/assets/102672238/edd378a6-cc6e-4810-9f9d-adcd938b14d8>
</p>
<p align="center"><em>
Overview of WhisperX’s alignment process <br/>
A similar process occurs for diarization
</em></p>
</br>

WhisperX was designed to work with a [faster-whisper](https://github.com/SYSTRAN/faster-whisper) backend model. This is a reimplementation of whisper that uses CTranslate2 to provide even faster transcription with lower GPU memory requirements. As such, before our model can be integrated with WhisperX, it must first be converted using CTranslate2. Luckily, this can be done easily using the Python [CTranslate2 library](https://opennmt.net/CTranslate2/guides/transformers.html#whisper).

### 2.7. Real-Time Transcription

An additional final functionality we wanted to achieve with our solution was real-time transcription. In comparison to asynchronous transcription, where the model receives the entire audio first before it begins transcription, real-time transcription involves the model receiving audio in real-time, and transcribing it ‘on-the-fly’. As such, the resultant transcription should be updated in real-time with new text being constantly appended, ideally in sync with the speaker.

In comparison to asynchronous transcription, where the user would first need to finish speaking and then wait for the model to finish processing before he can see the transcription, real-time transcription allows the user to transcribe his speech in real-time with minimal delay. Of course, the trade-off of this would be a lower accuracy of the real-time transcription, as the model does not have the full context of the entire audio file before it needs to make a prediction.

Whisper does not natively support real-time transcription, so we will likely need to explore some work-arounds to achieve this. The solution I ultimately settled on is to use multithreading to record audio in parallel with the transcription process. So essentially, there is one thread in the background that is constantly listening for audio and writing it in real-time to an internal buffer using PyAudio. On the other hand, the main thread will keep reading audio data from this buffer and send it to be transcribed by whisper. This successfully achieves pseudo real-time transcription.

However, one key issue with this is that when used for longer periods of transcription, the internal buffer will keep growing in size, and the whisper model will take longer and longer to transcribe it. This would cause the lag between the speaker and the transcription to increase rapidly, defeating the point of real-time transcription.

One possible solution to this is to naively clear the audio buffer every so often once it hits a certain limit. While this would work, this may cause the buffer to get cleared when the speaker is in the middle of saying a word. This would essentially cause the new buffer to start with a ‘partial’ word. Whisper may not be able to accurately transcribe this, which may cause it to hallucinate, reducing the quality of the rest of the transcription.

To address this, I have implemented a simple ‘smart-clearing’ system for the buffer. Essentially, once a buffer hits a certain length (soft_chunk_limit), the thread will look for the next period of silence before it clears the buffer. This helps reduce the likelihood of clearing the buffer in the middle of a word. However, in this case, it is still crucial to have another larger size limit (hard_chunk_limit) for the buffer that clears it immediately once reached. This would ensure that that even if the thread fails to detect any silence (e.g., due to a very noisy background), the buffer would still be cleared eventually in the worst case.

While this does work, this is still rather simplistic, as it only uses the energy level of the audio input to detect silence. This can be enhanced further in the future using a more robust VAD system that can specifically identify gaps between sentences, rather than just any silence.

## 3. Results

Word Error Rate (WER) when tested on an unseen sample of audio files consisting of a mix of Singaporean and international speakers:

•	Pre-trained foundational small model:  19.88%  
•	Our Finetuned model: 7.89%  

As seen, our finetuned showed a significant improvement in accuracy when transcribing both Singaporean and international speakers. Its error rate was less than half of that of the foundational model.

Next, we explored the time taken by our model to transcribe ~140 minutes of audio when used on different computational devices. This was to get a gauge of how the model performs in real-world use cases. We chose a 140-minute audio sample, as it was longest audio file we had from our datasets.

</br>
<p align="center">
  <img src=https://github.com/Nsohko/WhisperSG/assets/102672238/e0bb04e0-bf1c-4e25-9411-8a56cf4d9567>
</p>
<p align="center"><em>
Finetuned Whisper’s performance across various computational devices
</em></p>
</br>

The first two rows correspond to the GPU and CPU of the DSTA development notebook respectively, while the next two rows correspond to the GPU and CPU of another pc with weaker specs.

As seen, when running on a GPU, the model performs extremely well. For both transcription alone, and transcription + alignment + diarization, the inference is relatively fast on both GPUs.

However, when tested on CPUs instead, the model performs much poorer, requiring much longer time than the GPUs. For transcription alone, the time taken is still somewhat reasonable, however for transcription + alignment + diarization, the process takes extremely long, especially on the older CPU.

## 4. Limitations

Some of the key limitations we identified are as follows:
1.	Dataset Quality  
•	Despite diluting it, the IMDA NSC’s high degree of repetition still posed a challenge. Parts 1 and 2 were also not as diverse in covering Singlish terms as we had hoped  
•	This limited the generalizability of the final model, especially when tested on out-of-distribution data  
•	One potential method of addressing this is by also using parts 3 – 6 of the IMDA NSC for finetuning. However, as mentioned previously, these are unsuitable in their current form due to a lack of proper capitalization and punctuation in their transcriptions. As such, we would first need to find a way to restore these to the transcriptions before using them for finetuning  
•	These could be done by first passing the transcriptions through another language model to restore capitalization and punctuation.   

2.	Storage Limitation  
•	During the pre-processing phase of the finetuning process, the raw audio files are first converted into floating point spectrograms  
•	These can get extremely large and eat up a lot of disc space. In fact, these spectrograms can even exceed over 10x the size of the initial audio files!  
•	These cause us to very quickly reach the storage limit of our notebook, preventing us from adding additional data for finetuning  
•	Eventually, if the goal is to finetune the whisper model on an even larger dataset, it is likely that this storage limitation will be the first problem that would need to be overcome.  


3.	Compute Power  
•	As mentioned previously, while our solution performs extremely well on GPUs, its performance still leaves a lot to be desired on CPUs  
•	This greatly limits the usability of this solution on devices with limited computational power or no GPU, including mobile phones  
•	This is likely a key challenge that needs to be solved before we can move onto real-world local deployment  

4.	Domain Specificity and Real-World Deployment  
•	While the model did show a significant improvement in performance on the test-data set, this may not necessarily correlate to an improvement during real-world use  
•	This is especially true in highly specific domains (e.g., during technical support calls), where the model may need to recognize words it has not seen before during training  
•	This could be addressed by finetuning the model using more domain-specific data that is more closely aligned with its use cases

## 5. Extensions & Learning Points
Ultimately, one of the most important factors to consider for any deep learning- project (including this one), is the availability of high-quality domain-specific data. As such, in the long-term, it will be imperative to further finetune this model using specific data that aligns more closely with its exact use cases to maximize its usability.

Furthermore, I think this model should also be explored more in a deployment setting – especially when being used locally on mobile phones to transcribe technical support calls. This will allow us to identify any new issues that pop up when the model is being used in the real-world, and it would also let us further improve our product to meet the specific needs of our users.

Beyond that, I think we should also explore how we could provide even greater functionality. Currently, the diarization model used for speaker identification is rather inaccurate and often makes poor predictions. This would likely be the first area that needs to be enhanced. Alternatively, we could also explore the integration of the model with a LLM to summarize transcribed conversations. We could even explore the integration of traditional audio augmentation techniques, e. g. implementing the ASR with noise cancellation capabilities to improve its usability even in noisy environments.

Lastly, we should also explore how we could further enhance the real-time transcription functionality, for instance, by providing compatibility with speaker diarization, or integrating a more robust silence detection mechanism.

## 6. Usage
The main code for WhisperSG is all hosted on this repo. The finetuned model is hosted on Huggingface [here](https://huggingface.co/Nsohko/whispersg_small.en). I have tried to add comments to as much of the code as I can to help self-document it.

WhisperSG is packaged as a python package using pip making it very easy to integrate with other Python projects. The code for finetuning the model is also provided inside the repo, allowing for future users to easily further finetune the model if necessary. Similarly, the code to download and pre-process parts 1, 2 and 3 of the IMDA NSC is also provided for reference.

A brief description of the project's structure:

```
./demo: consists of the demo scripts for asynchronous and real-time (live) transcription. Can be used directly by the Python interpreter, or via command prompt (see the Inference section below)

./finetuning: scripts to load the IMDA dataset, preprocess it and finetune the model

./models: empty directory to store the downloaded finetuned whispersg model. This model will be automatcially downloaded the first time whispersg is invoked

./output: default directory to store results during inference

./whispersg: contains the actual library code of whispersg
```

</br>
WhisperSG has been tested on Python 3.10, and on the following specs:

CPU: Intel i9-12900H; GPU: RTX3080 (16GB)  
CPU: Intel i5-7500; GPU: GTX1060Ti (6GB)  

Please see below for further details on setting up, and using WhisperSG

### 6.1. Setup

Creating a new conda environment is strongly recommended due this project's large number of version-specific dependencies.

```
conda create --name whispersg python=3.10`  
conda activate whispersg
```

Next, we shall install pytorch together with compiled CUDA binaries from conda. Ensure that the version of torch and CUDA installed is compatible with your specific device. WhisperSG has been tested using pytorch==2.3.0 and CUDA==12.1.  
`conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`

Next install ffmpeg through conda.  
`conda install ffmpeg=4.2.2`

Clone into this repository, and move into it.
```
git clone https://github.com/Nsohko/whisperSG
cd whispersg
```

Install this repo as an editable package. This step may take a while.  
`pip install -e .`

Finally, login into huggingface using the following command. This will prompt you to input your huggingface token.  
`huggingface-cli login`

For finetuning, please also create a new ipykernel in the environment that can be used to run the finetuning jupyter notebook:  
`python -m ipykernel install --user --name gpu_env --display-name "WhisperSG (GPU)"`

### 6.2. Inference

WhisperSG can be easily invoked via command line from inside the conda environment. It will automatically download the finetuned model from huggingface and use that by default. If the finetuned model is not available for whatever reason, it will default to the foundational small.en model

For asynchronous transcription `./demo/transcribe.py` will be used. Please enter:  
`whispersg`

For real-time transcription, `./demo/live_transcribe.py` will be used. Please enter:  
`whispersg_live` 

Both whispersg and whispersg_live have a large list of optional arguments to modify their behavior. Please enter `whispersg -h` and `whispersg_live -h` for the full list of arguments, and documentation on what they do.

Note that translation is also supported, by passing the `--task translate` argument. However, in this case, please also change the model argument to a multilingual model (for instance, by passing `--model small`). This is because otherwise whispersg defaults to the finetuned English-only model, which does not support translation, only English transcription. It is also recommended to pass in the input language using the `--langauge` argument.

So for example, to translate from Chinese to English, please enter the following commands  
For asynchronous: `whispersg --model small --task translate --language Chinese`  
For real-time: `whispersg_live --model small --task translate --language Chinese`  

If doing inference on CPU or older GPUs that do not support efficient float16 compuation, please also pass in the following argument: `--compute_type int8`

For python usage, please make use of WhisperSGPipeline class stored under `./whispersg/pipeline.py`. This takes in essentially the same arguments as the `whispersg` command (which runs `./demo/transcribe.py`). The pipeline is also used in `./demo/live_transcribe.py`

During inference, if it is observed that the model keeps 'missing out' on certain chunks of audio, this is likely an issue with the VAD being too sensitive and excessively filtering out audio. To address this, please reduce the `vad_onset` and `vad_offset` parameters. This can be done by either passing the corresponding arguments to `whispersg` or passing in the new values during the initialisation of a WhisperSGPipeline object.

For more details on usage / bugs, please also see [here](https://github.com/m-bain/whisperX)

### 6.3. Finetuning

#### 6.3.1 Downloading the dataset

The IMDA dataset is stored on dropbox, as such you will likely need a dropbox premium account to access it. Dropbox currently has a 30 day free trial available. You must then also download the Dropbox desktop app.

After requesting access to the IMDA NSC [here](https://docs.google.com/forms/d/e/1FAIpQLSd3k8wFF4GQP4yo_lDAXKjCltfYk-dE-yYpegTnCB20kr7log/viewform), you should be able to access it from the dropbox after a few days. From there, I strongly recommend choosing the parts you require, and making them available offline.

Next, we will need to do some early pre-processing to extract out and cut the audio files, and lightly format the transcriptions as needed. I have provided scripts to do this for parts 1,2 and 3 under `./finetuning/data_preprocessing/`

For parts 1 and 2, please use the `load_part1or2.py` script, while for part3 use the `load_part3.py` script. Please adjust the arguments in the args_class at the top of the scripts as necessary.

These scripts will generate two .txt files as listed below. These will be used for the next step of the pre-processing. If using another custom dataset other than the IMDA NSC, you will need to generate these two .txt files yourself in the same format.

1. `audio_paths.txt`: Consists of pairs of unique audio IDs and the absolute file path to the corresponding audio files
```
<unique_id>   <absolute path to the audio file-1>
<unique_id>   <absolute path to the audio file-2>
...
<unique_id>   <absolute path to the audio file-N>
```

2. `text.txt`: Contains the transcriptions corresponding to each of the audio files mentioned in audio_paths.txt. The transcriptions should be indexed by its corresponding audio file’s unique audio ID, as stored in audio_paths.txt. The order of IDs in audio_paths.txt and text.txt should also be consistent
```
<unique_id>   <Transcription (ground truth) corresponding to the audio file-1>
<unique_id>   <Transcription (ground truth) corresponding to the audio file21>
...
<unique_id>   <Transcription (ground truth) corresponding to the audio file-N>
```

After obtaining these two .txt files, we will next need to save the datasets as .arrow chunks.These can be done using the `finalize_data.py` script, also located in `./finetuning/data_preprocessing/`. Again please pass in the required arguments at the top of the script in the args_class. This script is a modfied version from [here](https://github.com/vasistalodagala/whisper-finetune). More details can be found there, including sample examples of how the .txt files should be formatted.

During this step, some light pre-processing is also carried out on the data. In particular, it removes all duplicate examples (keeping only 1 randomly chosen sample), and removes extra transcriptions that do not have any corresponding audio_file.

#### 6.3.2 Finetuning

As mentioned previously, the finetuning process is largely streamlined by the Huggingface transformers Sequence-to-Sequence trainer. I have placed the jupyter notebook used for finetuning under `./finetuning/training`. Please ensure that the notebook is running on the WhisperSG (GPU) kernel created previously during setup.

As with before, the paths to the local datasets & training hyperparameters can be easily modified as needed by adjusting the arguments at the top of the script in the args_class. Right now, the notebook is primarily only intended for use when finetuning with local datasets. For examples of finetuning notebooks for huggingface datasets instead, please see [here](https://github.com/vasistalodagala/whisper-finetune).

Before the actual finetuning process, the notebook first downloads the foundational Whisper model. It then further preprocesses and augments the datasets. During this data processing stage (which uses the Huggingface datasets map function), occasionaly an error along the following lines pop up: `OS Permission Denied`. This can be safely ignored, and simply rerunning the notebook cell should fix it. Also note that it is during this stage where the raw audio files are converted into spectrograms. These spectrograms take up an extremely large amount of space (10+ times the size of the original audional files), so please ensure that you have sufficient space on your hard drive. Other than that, most other warnings in the notebook can be usually safely ignored.

After completing the data preprocessing successfully, the notebook then saves the foundational whisper tokenizer in the output directory. After this, it will then initiate the finetuning process, during which it will save checkpoints of the finetuned model in the output directory, with a frequency based on the arguments provided earlier. Note that it appears that there is currently a bug with the huggingface trainer, where it only saves checkpoints of the finetuned model, without its tokenizer. To address this, we can simply copy the tokenizer files that were saved earlier to the root output directory into the model checkpoint's folder. This is safe as the finetuning process does not modify the tokenizer. 

</br>
<p align="center">
  <img src=https://github.com/Nsohko/WhisperSG/assets/102672238/8fd75d77-6d85-4d78-820e-19d41f71f0a8)>
</p>
<p align="center"><em>
Sample screenshot of output directory root
</em></p>
</br>

As shown in the image above, please copy the tokenizer files (`added_tokens.json, merges.txt, normalizer.json, special_tokens_map.json, tokenizer_config.json, vocab.json`) from the root of the output directory into the checkpoint folder. DO NOT copy `pre_processor_config.json` (unless it is also missing), as the trainer should have already created one inside the checkpoint folder. Similarly, do not copy any other file that would overwrite a file of the same name inside the checkpoint folder.

At the end, the checkpoint folder should have the following files:

</br>
<p align="center">
  <img src=https://github.com/Nsohko/WhisperSG/assets/102672238/f0d8e120-976a-4df9-99a1-13c94e352f5b>
</p>
<p align="center"><em>
Final checkpoint folder
</em></p>
</br>

#### 6.3.3 Integration

After finetuning our model, there are a few more steps we need to do before we can integrate it with WhisperX and the rest of WhisperSG. This is because, as mentioned previously, WhisperX expects a faster-whisper backend model, so we will first need to convert our model using CTranslate2.

This can be done easily using the following command from within the WhisperSG conda environment. Please replace the path/to/finetuned/checkpoint and path/to/output/dir as necessary:  
`ct2-transformers-converter --model path/to/finetuned/checkpoint --output_dir path/to/output/dir`

After that, the path to the resulting model can be provided under the `--model` argument for WhisperSG









