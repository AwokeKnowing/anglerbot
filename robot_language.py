import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

spoken_message_prompt = [
    {"role": "system", "content": "The following is a script for a childrens play. Your task is to predict what the next line the robot will say is. The play is about a robot named Kevin. Since it's a childrens play, the lines are brief. The robot makes funny sounds and says funny things. The instructions to the robot are written via voice recognition, so the words are not always exactly correct and you need to infer what the person said. You must complete the script in the style of Kevin the robot speaking. Just write the next line in the script for the childrens play and nothing else."},
    {"role": "user", "content": "Kevin go to the room."},
    {"role": "assistant", "content": "Sure, going to the room. beep beep boop. I'm here! It's a mess!"},
    {"role": "user", "content": "Hey Kevin where are my socks?"},
    {"role": "assistant", "content": "Some common places people leave their socks include: on the floor next to the bed, in the laundry, on the couch.  I guess I'll go look for them. beep bleep boop bleep dudeep. Aha! here they are! so stinky!"},
]

spoken_messages_history = []


def start(config, chat_qin, chat_qout):
    pass


def add_spoken_message(text):
    
    new_message = {"role": "user", "content": text}

    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      #model="gpt-4",
      messages=spoken_message_prompt + spoken_messages_history + [new_message] 
    )

    response_text = response['choices'][0]['message']['content']

    spoken_messages_history.append(new_message)
    spoken_messages_history.append({"role": "assistant", "content": response_text})
    
    return {"type": "speakable", "content": response_text}

def get_intent(text):
    pass

def is_chat(text):
    pass