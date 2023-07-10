#! python3.7

import argparse
import multiprocessing as mp
import threading
import time

import cowsay
import whitematter

import robot_hearing
import robot_language
import robot_voice
import robot_config


class Brain:

    robot_queue_types = {
        "/hearing/statement": "string",
        "/language/chat/in/statement": "string",
        "/language/chat/out/statement": "string",
        "/voice/statement": "string",
    }

    robot_processors = {
        "robot_hearing": robot_hearing.start,
        "robot_language": robot_language.start,
        "robot_voice": robot_voice.start,
    }

    def wake(self):
        with robot_config.Configuration("robot_config.json") as config:
            self.is_being_addressed=False
            self.robot_asked_question=False

            self.whiteFiber = whitematter.WhiteFiber(self.robot_queue_types.keys())

            self.axon=self.whiteFiber.axon(
                in_topics=[
                    "/hearing/statement",
                    "/language/chat/out/statement"
                ],
                out_topics=[
                    "/language/chat/in/statement",
                    "/voice/statement"
                ]

            )
            
            

            # Create a process for the worker functions
            for pname, pstart in self.robot_processors.items():
                self.robot_processors[pname] = threading.Thread(target=pstart, args=(config,self.whiteFiber), daemon=True)
                self.robot_processors[pname].start()

            # Read messages from the queue
            while True:
                try:
                    # Block until a message is available in the queue
                    message = self.axon["/hearing/statement"].get()
                    
                    # Do something with the message
                    if len(message)>0 and message != " " and message != ".":
                        cowsay.cow(message)
                    if message.lower().startswith("turn yourself off") and len(message)<19:
                        cowsay.cow("Goodbye!")
                        break

                    words = ''.join([c for c in message if c not in '¡!¿?,.']).lower().split()
                    print("words", words)
                    words_to_robot = self.get_words_addressed_to_robot(words)
                    if len(words_to_robot) > 1 or self.robot_asked_question:
                        print("Thinking...")
                        self.axon["/language/chat/in/statement"].put(message)
                        response = self.axon["/language/chat/out/statement"].get()
                        
                        self.robot_asked_question = response[-1]=="?"
                        cowsay.tux(response)
                        self.axon['/voice/statement'].put(response)
                        
                except KeyboardInterrupt:
                        cowsay.cow("Goodbye!")
                        break

    def get_words_addressed_to_robot(self, words, name='kevin', before=8):
        try:
            name_index = words.index(name)
            start_index = max(0, name_index - before)
            return words[start_index:]
        except ValueError:
            return []


if __name__ == "__main__":
    Brain().wake()