#! python3.7

import argparse
import multiprocessing as mp
import time

import cowsay

import robot_hearing
import robot_language
import robot_voice
import robot_config



class Brain:

    robot_queues = {
        "/hearing/in/statements": None,
        "/language/out/chat": None,
        "/language/in/chat": None,
        "/voice/out/speech": None
    }

    is_being_addressed=False
    robot_asked_question=False
    def get_words_addressed_to_robot(self, words, name='kevin', before=8):
        try:
            name_index = words.index(name)
            start_index = max(0, name_index - before)
            return words[start_index:]
        except ValueError:
            return []

    def wake(self):
        with robot_config.Configuration("robot_config.json") as config:
        

            # Create a queue for the messages
            self.robot_queues["/hearing/in/statements"] = mp.Queue()
            self.robot_queues["/language/out/chat"] = mp.Queue()
            self.robot_queues["/language/in/chat"] = mp.Queue()
            self.robot_queues["/voice/out/chat"] = mp.Queue()

            # Create a process for the worker function
            hearing_process = mp.Process(target=robot_hearing.start, args=(config,self.robot_queues["/hearing/in/statements"]))
            hearing_process.start()

            # Read messages from the queue
            while True:
                try:
                    # Block until a message is available in the queue
                    print("waiting for statement")
                    message = self.robot_queues["/hearing/in/statements"].get()
                    print("got statement")
                    # Do something with the message
                    if len(message)>0 and message != " " and message != ".":
                        cowsay.cow(message)
                    if message.lower().startswith("turn yourself off") and len(message)<19:
                        cowsay.cow("Goodbye!")
                        hearing_process.terminate()
                        break

                    words = ''.join([c for c in message if c not in '¡!¿?,.']).lower().split()
                    print("words", words)
                    words_to_robot = self.get_words_addressed_to_robot(words)
                    if len(words_to_robot) > 1 or self.robot_asked_question:
                        print("Thinking...")
                        response = robot_language.add_spoken_message(message) 
                        
                        if response["type"] == "speakable":
                            self.robot_asked_question = response['content'][-1]=="?"
                            cowsay.tux(response["content"])
                            robot_voice.speak(response["content"])
                    
                    
                except KeyboardInterrupt:
                        cowsay.cow("Goodbye!")
                        hearing_process.terminate()
                        break



if __name__ == "__main__":
    Brain().wake()