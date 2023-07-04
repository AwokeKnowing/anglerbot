from time import sleep
from datetime import datetime, timedelta
from queue import Queue
import io
from tempfile import NamedTemporaryFile
import whisper
import speech_recognition as sr
from sys import platform
import torch


def start(config, messages_queue):
    source=None
    # The last time a recording was retreived from the queue.
    phrase_time = None
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = config['ears.mic_energy_min']
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False
    
    # Prevents permanent application hang and crash by using the wrong Microphone
    mic_name = config['ears.mic_device']
    if not mic_name or mic_name == 'list':
        print("Available microphone devices are: ")
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"Microphone with name \"{name}\" found")   
        return
    else:
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            if mic_name in name:
                source = sr.Microphone(sample_rate=16000, device_index=index)
                print ("mic",source)
                break
    
        
    # Load / Download model
    model = config['ears.model']
    if model != "large" and not config['ears.multilingual']:
        model = model + ".en"
    audio_model = whisper.load_model(model)

    record_timeout = config['ears.record_timeout']
    phrase_timeout = config['ears.phrase_timeout']

    temp_file = NamedTemporaryFile().name
    transcription = ['']
    
    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    print("Model loaded.\n")
    have_sent_message=False


    while True:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            # if it's empty, do nothing
            # but if empty for a while, send the last recorded text
            if data_queue.empty():
                # if phrase is over and we haven't yet sent back the phrase, send it (don't wait til we get next bytes)
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout) and not have_sent_message:
                    messages_queue.put(transcription[-1])
                    have_sent_message=True
                continue

            have_sent_message = False
            phrase_complete = False
            # If enough time has passed between recordings, consider the phrase complete.
            # Clear the current working audio buffer to start over with the new data.
            if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                last_sample = bytes()
                phrase_complete = True
            # This is the last time we received new audio data from the queue.
            phrase_time = now

            # Concatenate our current audio data with the latest audio data.
            while not data_queue.empty():
                data = data_queue.get()
                last_sample += data

            # Use AudioData to convert the raw data to wav data.
            audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
            wav_data = io.BytesIO(audio_data.get_wav_data())

            # Write wav data to the temporary file as bytes.
            with open(temp_file, 'w+b') as f:
                f.write(wav_data.read())

            # Read the transcription.
            result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available())
            text = result['text'].strip()
            #print(text)

            # If we detected a pause between recordings, add a new item to our transcripion.
            # Otherwise edit the existing one.
            if phrase_complete:
                transcription.append(text)
            else:
                transcription[-1] = text

            # Clear the console to reprint the updated transcription.
            #os.system('cls' if os.name=='nt' else 'clear')
            #for line in transcription:
            #    print(line)
            
            # Flush stdout.
            #print('', end='', flush=True)

            # Infinite loops are bad for processors, must sleep.
            sleep(0.25)
        except KeyboardInterrupt:

            break

    #print("\n\nTranscription:")
    #for line in transcription:
    #    print("Heard: " + line)

    #messages_queue.put("Go to sleep")
