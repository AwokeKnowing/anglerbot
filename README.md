# Kevin


This is a demo of real time speech to text with OpenAI's Whisper model. It works by constantly recording audio in a thread and concatenating the raw bytes over multiple recordings.

To install dependencies simply run
```
python3 -m venv venv
source venv/bin/activate
sudo apt install python3-pyaudio portaudio19-dev ffmpeg
pip install -r requirements.txt
```
in an environment of your choosing.


For more information on Whisper please see https://github.com/openai/whisper

The code in this repository is public domain.


        sense   (sensor input)
      perceive  (deep learning encode)
    emote    (calculate internal low level motivations like battery, connect, help )
  concern       (attention over world model with possible futures)
trust         (core policy. score futures. select goal)
  act     (evaluate actions to acheive goal)
    try      (select behavior)
      orchestrate    (track progress and emit actions)
        react     (actuators output)


def invert()
spectator


tick
lambda: #Root
  lambda: #Chooser
    SensePainCondition() and HandlePainAction() or\
    IsMovingCondition()  and  or\
    FindMove()
