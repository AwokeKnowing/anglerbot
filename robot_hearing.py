#!/usr/bin/env python3

import threading
import speech_recognition as sr
import time
import sounddevice  #gets rid of all the ALSA messages printed to console (just importing it does that)
import copy
import io


def start(config, wf):
    source=None
    mic=None

    axon=wf.axon(
        in_topics=[

        ],
        out_topics=[
            "/hearing/statement"
        ]
    )
    
    # obtain audio from the microphone
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = config['ears.mic_energy_min']

    # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
    #r.dynamic_energy_threshold = False
    #r.pause_threshold=2

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
                mic = sr.Microphone(sample_rate=16000, device_index=index)
                
                print ("mic",repr(source))
                break

    #with mic as source: r.adjust_for_ambient_noise(source)

    def transcribe(axon, recognizer, audio, mic):
        try:
            audio2 = copy.deepcopy(audio)
            print("transcribing")
            text = recognizer.recognize_google(audio2,key=None,language="en-US",pfilter=1,show_all=False,with_confidence=False)
            if len(text):
                axon["/hearing/statement"].put(text)
        except sr.UnknownValueError:
            print("didn't understand")
            
        except sr.exceptions.UnknownValueError:
            print("really didn't understand")
            
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

    while True:
        try:
            print("Kevin is listening...")
            with mic as source:audio = recognizer.listen(source, timeout=20,phrase_time_limit=12)

            transcriber = threading.Thread(target=transcribe, args=(axon, recognizer, audio, mic))
            transcriber.start()
        
        except sr.WaitTimeoutError:
            with mic as source: recognizer.adjust_for_ambient_noise(source)
            print("heard nothing")

        time.sleep(.01)
