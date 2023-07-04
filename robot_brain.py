#! python3.7

import argparse
import multiprocessing as mp
import time

import cowsay

import robot_hearing
import robot_language
import robot_voice
import robot_config

robot_queues = {
    "/hearing/in/statements": None,
    "/language/out/chat": None,
    "/language/in/chat": None,
    "/voice/out/speech": None
}



def main():
    with robot_config.Configuration("robot_config.json") as config:
    

        # Create a queue for the messages
        robot_queues["/hearing/in/statements"] = mp.Queue()

        # Create a process for the worker function
        hearing_process = mp.Process(target=robot_hearing.start, args=(config,robot_queues["/hearing/in/statements"]))
        hearing_process.start()

        # Read messages from the queue
        while True:
            try:
                # Block until a message is available in the queue
                message = robot_queues["/hearing/in/statements"].get()

                # Do something with the message
                if len(message)>0 and message != " " and message != ".":
                    cowsay.cow(message)
                if message.lower().startswith("turn yourself off") and len(message)<19:
                    cowsay.cow("Goodbye!")
                    hearing_process.terminate()
                    break;

                words = ''.join([c for c in message if c not in '¡!¿?,.']).lower().split()
                print(words)
                #if (len(words) >= 1 and (words[0] == 'charlie' or words[0] == 'charles')) or \
                #   (len(words) >= 2 and (words[1] == 'charlie' or words[1] == 'charles')) or \
                #   (len(words) >= 3 and ((words[1]=="right" or words[1]=="ok") and (words[2] == 'charlie' or words[2] == 'charles'))):
                #    print("Thinking...")
                if (len(words) >= 1 and (words[0] == 'kevin' or words[0] == 'kevin')) or \
                   (len(words) >= 2 and (words[1] == 'kevin' or words[1] == 'kevin')) or \
                   (len(words) >= 3 and ((words[1]=="right" or words[1]=="ok") and (words[2] == 'kevin' or words[2] == 'kevin'))):
                    print("Thinking...")
                    response = robot_language.add_spoken_message(message) 

                    if response["type"] == "speakable":
                        cowsay.tux(response["content"])
                        robot_voice.speak(response["content"])
                   
                
            except KeyboardInterrupt:
                    cowsay.cow("Goodbye!")
                    hearing_process.terminate()
                    break



if __name__ == "__main__":
    main()