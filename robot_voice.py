import boto3
import simpleaudio as sa

import robot_voice_effect

robot=True
voice="Matthew"

polly = boto3.client('polly', region_name='us-west-2')


def speak(text):

    #text = "<speak>Hi! I'm Matthew. Hope you are doing well. This is a sample PCM to WAV conversion for SSML. I am a Neural voice and have a conversational style. </speak>" # Input in SSML

    frames = []
    try:
        if "<speak>" in text: #Checking for SSML input
            #Calling Polly synchronous API with text type as SSML
            response = polly.synthesize_speech(Text=text, TextType="ssml", OutputFormat="pcm",VoiceId=voice, SampleRate="16000") #the input to sampleRate is a string value.
        else:
             #Calling Polly synchronous API with text type as plain text
            response = polly.synthesize_speech(Text=text, TextType="text", OutputFormat="pcm",VoiceId=voice, SampleRate="16000")
    except (BotoCoreError, ClientError) as error:
        print(error)
        
    #print(response)
    #Processing the response to audio stream
    stream = response.get("AudioStream")
    frames.append(stream.read())

    if robot:
        frames = robot_voice_effect.from_pcm(frames)
    
    audio= b''.join(frames)

    play_obj = sa.play_buffer(audio, 1, 2, 16000)
    play_obj.wait_done()#robot shouldn't ever play more than one voice at once