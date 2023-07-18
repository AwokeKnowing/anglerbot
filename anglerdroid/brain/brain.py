#! python3.7

import argparse
import multiprocessing as mp
import threading
import itertools
import time

import cowsay

import anglerdroid as a7

class Brain:
    def __init__(self,configJson):
        self.configJson=configJson

        self.topic_defs = {
            "/hearing/statement": "string",
            "/language/chat/in/statement": "string",
            "/language/chat/out/statement": "string",
            "/voice/statement": "string",
            "/gamepad/diffdrive/leftrightvels":"tuple",
            "/wheels/diffdrive/leftrightvels":"tuple",
            "/vision/images/topdown":"BGRImage"
        }

        self.sensors = {
            "hearing": a7.hearing.start,
            "gamepad": a7.gamepad.start,
            "vision":  a7.vision.start,
        }
        self.actuators = {
            "voice": a7.voice.start,
            "wheels": a7.wheels.start,
        }
        self.processors = {
            "language": a7.language.start,
        }

        self.handlers = {}

        self.brainSleeping = threading.Event()

    def wake(self):
        print("waking up")
        with a7.Configuration(self.configJson) as robotConfig:
            self.is_being_addressed=False
            self.robot_asked_question=False

            self.whiteFiber = a7.WhiteFiber(self.topic_defs.keys())

            self.axon=self.whiteFiber.axon(
                get_topics=[
                    "/hearing/statement",
                    "/language/chat/out/statement",
                    "/gamepad/diffdrive/leftrightvels"
                ],
                put_topics=[
                    "/language/chat/in/statement",
                    "/voice/statement",
                    "/wheels/diffdrive/leftrightvels",
                ]

            )
            

            # Create a process for the worker functions
            for pname, pstart in itertools.chain(self.sensors.items(),self.actuators.items(),self.processors.items()):
                self.handlers[pname] = threading.Thread(target=pstart, args=(robotConfig,self.whiteFiber,self.brainSleeping))
                self.handlers[pname].daemon = True
                self.handlers[pname].start()

            cowsay.tux("Hello!")

            # Read messages from the queue
            while not self.brainSleeping.isSet():
                try:
                    # Block until a message is available in the queue
                    message = self.axon["/hearing/statement"].get()
                    wheelVels = self.axon["/gamepad/diffdrive/leftrightvels"].get()
                    chatToSay = self.axon["/language/chat/out/statement"].get()
                    
                    if wheelVels is not None:
                        left,right = wheelVels
                        if left or right:
                            print("wheels",wheelVels)

                        self.axon["/wheels/diffdrive/leftrightvels"].put(wheelVels)

                        
                    
                    # Do something with the message
                    if message:
                        if len(message)>0 and message != " " and message != ".":
                            cowsay.cow(message)
                        if message.lower().startswith("turn yourself off") and len(message)<19:
                            break

                        words = ''.join([c for c in message if c not in '¡!¿?,.']).lower().split()
                        print("words", words)
                        words_to_robot = self.get_words_addressed_to_robot(words)
                        if len(words_to_robot) > 1 or self.robot_asked_question:
                            print("Thinking...")
                            self.axon["/language/chat/in/statement"].put(message)
                            

                    if chatToSay:
                        self.robot_asked_question = chatToSay[-1]=="?"
                        cowsay.tux(chatToSay)
                        self.axon['/voice/statement'].put(chatToSay)

                    time.sleep(.0001)
                        
                except KeyboardInterrupt:
                    cowsay.tux("Goodbye!")
                    break
            
            self.shutdown()
            cowsay.tux("Zzzz")
                    
    def shutdown(self):
        self.brainSleeping.set()
        time.sleep(5)
        for name,handler in self.handlers.items():
            if handler.isAlive():
                print("waiting for "+ name)
                handler.join(timeout=20)
                if handler.isAlive():
                    print("force stopping "+name)

    def get_words_addressed_to_robot(self, words, name='kevin', before=8):
        try:
            name_index = words.index(name)
            start_index = max(0, name_index - before)
            return words[start_index:]
        except ValueError:
            return []
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("sleeping")


if __name__ == "__main__":
    Brain().wake()