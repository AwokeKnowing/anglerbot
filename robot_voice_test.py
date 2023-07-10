import boto3
import robot_voice_effect
import simpleaudio as sa #must be imported in process that uses

def start(config, rq):
    
    robot=False
    voice="Matthew"

    polly = boto3.client('polly', region_name='us-west-2')
    #text = "<speak>Hi! I'm Matthew. Hope you are doing well. This is a sample PCM to WAV conversion for SSML. I am a Neural voice and have a conversational style. </speak>" # Input in SSML

    while True:
        text = rq['/voice/statement'].get()
        frames = []
        try:
            if "<speak>" in text: #Checking for SSML input
                #Calling Polly synchronous API with text type as SSML
                response = polly.synthesize_speech(Text=text, TextType="ssml", OutputFormat="pcm",VoiceId=voice, SampleRate="16000") #the input to sampleRate is a string value.
            else:
                #Calling Polly synchronous API with text type as plain text
                response = polly.synthesize_speech(Text=text, TextType="text", OutputFormat="pcm",VoiceId=voice, SampleRate="16000",Engine="neural")
        except (boto3.BotoCoreError, boto3.ClientError) as error:
            print(error)
            continue
            
        #print(response)
        #Processing the response to audio stream
        stream = response.get("AudioStream")
        audio=stream
        print("audio", repr(audio))
        if robot:
            frames.append(stream.read())

            if robot:
                frames = robot_voice_effect.from_pcm(frames)
            
            audio= b''.join(frames)

            play_obj = sa.play_buffer(audio, 1, 2, 16000)
            play_obj.wait_done()#robot shouldn't ever play more than one voice at once
        else:
            play_obj = sa.play_buffer(audio.read(), 1, 2, 16000)
            play_obj.wait_done()#robot shouldn't ever play more than one voice at once
