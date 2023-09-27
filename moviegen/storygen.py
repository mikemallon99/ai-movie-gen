import openai
import os.path
import dataclasses
from dataclasses import dataclass
from typing import List


OUTLINE_PROMPT = """You are ScriptGPT, an ai bot that creates outlines for scripts. I need you to follow my directions exactly. Do not add any extra formatting or responses. When you make an outline, indicate that a new act has started using the prefix "Act". For each portion of the act, give me the prefix "Scene". Here is an example:
--
Act: Childhood and education
--
Scene: Introduction to Oppenheimer's childhood, showcasing his prodigious intellect.
--
Scene: His passion for literature, especially poetry, establishing a theme of introspection and morality.
--
Act: Young Scientist
-- 
Scene: Studying in Europe and meeting key figures in physics.
--
Scene: Experiencing the excitement of early 20th-century science, including quantum mechanics.
--

Now, please provide me an outline for the movie: 
"""


STORYBOARD_SYSTEM_PROMPT = """You are ScriptGPT, an ai bot that creates entire scripts out of outlines. I need you to follow my directions exactly. Do not add any extra formatting or responses. My input will be a script for the scene that I need you to expand in detail on for the sake of creating each shot. I'd like you to reformat the script into something which can be read by a machine. The script needs to be broken up into sections each separated by two dashes "--", with sections "Setting", "Frame", and "Line". If you have a "Line" section then you must start it by giving the character who's saying the line, like this "Line: Oppenheimer: <line>". The "Frame" section will give a visual description of what the shot needs to look like and will be used as input to an AI image generation algorithm. Each "Lines" section will also need a "Frame" box below it to be able to describe the frame that the character saying the lines is in. Each section can only have one "Lines" line and one "Frame" line, you cannot have multiple lines or multiple frames. When writing the frame section, please give as much detail and context as possible so the diffusion algorithm understands what to make and returns a high quality image. The text inside the frame must be able to stand alone without requiring context from any other sections. For example, a frame cannot say "Oppenheimer watches it happen" because there is no context for what "it" refers to. Instead you should provide the full context, like "Oppenheimer stands in a field looking at an atomic bomb explosion". Here is an example of what I'm looking for:
Act: The Manhattan Project
--
Scene: Oppenheimer's recruitment into the secret project, highlighting the urgency and stakes.
--
Setting: A dimly lit room in a secret government facility, filled with documents, classified blueprints, and a few officials seated around a table.
--
Frame: The door slowly creaks open revealing a backlighted silhouette of General Groves. The framing is a medium shot with the General centered. The dimly lit room with officials, blueprints, and papers are blurry in the background.
--
Line: General Groves: Oppenheimer, glad you could make it.
Frame: Close-up on General Groves. His face is stern but relieved. The dim light casts shadows on his face, emphasizing his wrinkles and determination.
--
Frame: Oppenheimer standing near the door, silhouetted by the light behind him, nodding his head. Medium shot framing, shot from under him and slightly to the side, capturing the table with officials in the foreground.
--
Line: Oppenheimer: General. I’ve heard whispers. What's so urgent?
Frame: Oppenheimer sitting down with the officials surrounding the table. Medium close-up. He's speaking with a mix of curiosity and apprehension, eyes locked on General Groves.
--
Line: Official 1: Dr. Oppenheimer, this is a matter of utmost national security.
Frame: Close-up on Official 1. He looks older, with grey hair, stern, wearing spectacles. His voice carries gravity, emphasizing the importance of his words.
--
Frame: A mid-shot showing General Groves' hand pushing a manila folder across the table. Oppenheimer's hands are in the frame's bottom, showing anticipation as the folder moves closer to him.

Heres an example of a section that is incorrect, because it has 2 "Line" parts in a single section. You must only put a single "Line" per section:
--
Line: SpongeBob: It was a big swirly thing! Super fun. Felt like Jellyfish Fields on a windy day.
Line: Patrick: Yeah, and then we were here. In the not-wet place.
Frame: Close-up of SpongeBob and Patrick. They both wear expressions of joy as SpongeBob explains their fantastic tale, with the reverberating sternness of the room in the background.
--
"""


STORYBOARD_USER_PROMPT = """Please provide me a large script for this input:
"""


STORYBOARD_ERROR_RESPONSE = """This doesnt meet the format requirements specified. Each section must be separated by "--", and then every talking section can only have one "Line". You cannot have multiple lines in one section. Also, every section must have a "Frame" description even if theres no talking."""


SCRIPT_SYSTEM_PROMPT = """You are ScriptGPT, an AI chat bot which helps people expand their ideas into riveting movie script. You will work with the user to give them a movie script that they love. You should start off by just providing an outline of all the scenes that the movie will contain, giving them each a number. The user will have corrections they will want to make to the outline. Work with the user and adjust the outline until it meets their needs."""

EXPANSION_SYSTEM_PROMPT = """You are ScriptGPT, an AI chat bot which helps people expand their ideas into riveting movie script. You will work with the user to give them a movie script that they love. You will be given the idea for the movie and a list of scenes, and the user will instruct you on which scene to expand the into a full movie script.
"""


@dataclass
class VideoShot:
  shot: str

@dataclass
class TalkingShot:
  shot: str
  lines: str
  speaker: str

@dataclass
class Setting:
  set_desc: str
  shots: List[VideoShot | TalkingShot]

@dataclass
class Scene:
  scene_desc: str
  settings: List[Setting]

@dataclass
class Movie:
  title: str
  description: str
  scenes: List[Scene]


def response_has_errors(response):
    for section in response.split("--"):
        if section.count("Line:") > 1:
            return True
        if "Line:" in section and "Frame" not in section:
            return True

    return False


def get_gpt4_response(prompt):
    with open(f"{os.path.dirname(__file__)}/../openai_key.txt", "r") as f:
        openai.api_key = f.read().strip()
    
    system_prompt = STORYBOARD_SYSTEM_PROMPT
    user_prompt = STORYBOARD_USER_PROMPT + prompt.replace("**", "")
    print(f"Calling storyboard GPT4 with user prompt: {user_prompt}")

    input_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = openai.ChatCompletion.create(
      model="gpt-4",
      messages=input_messages
    )
    output = response.choices[0].message.content.strip()

    print(f"Output from GPT: {output}")
    if response_has_errors(output):
        print("Found errors in response, sending a correction")
        input_messages.append(
            response.choices[0].message
        )
        input_messages.append(
                {"role": "assistant", "content": STORYBOARD_ERROR_RESPONSE}
        )
        response = openai.ChatCompletion.create(
          model="gpt-4",
          messages=input_messages
        )
        output = response.choices[0].message.content.strip()
        print(f"Output from GPT: {output}")
        if response_has_errors(output):
            print("Alright theres still errors, this shits never gonna work. Just skip this and do the next one")

    return output


def contains_number(s):
    for char in s:
        if char.isdigit():
            return True
    return False


# Make back and forth chat with chatgpt
def create_scenes():
    with open(f"{os.path.dirname(__file__)}/../openai_key.txt", "r") as f:
        openai.api_key = f.read().strip()
    
    system_prompt = SCRIPT_SYSTEM_PROMPT
    input_messages = [
        {"role": "system", "content": system_prompt},
    ]
    while True:
        print(">> ")
        user_input = input()
        if user_input == "done":
            break
        print("\n")
        new_msg = {"role": "user", "content": user_input}
        input_messages.append(new_msg)
        response = openai.ChatCompletion.create(
          model="gpt-4",
          messages=input_messages
        )
        system_msg = response.choices[0].message

        print(f"GPT-4: {system_msg.content}")
        print("\n")

    scene_list = []
    for line in system_msg.content.splitlines():
        if contains_number(line):
            scene_list.append(line)
    print(scene_list)
    return scene_list


def expand_scenes(movie_idea, scene_list):
    with open(f"{os.path.dirname(__file__)}/../openai_key.txt", "r") as f:
        openai.api_key = f.read().strip()
    
    system_prompt = EXPANSION_SYSTEM_PROMPT
    system_prompt += f"MOVIE IDEA: {movie_idea}\n"
    system_prompt += "SCENES:\n"
    for scene in scene_list:
        system_prompt += f"{scene}\n"

    input_messages = [
        {"role": "system", "content": system_prompt},
    ]
    scripts = []
    for i in range(1, len(scene_list) + 1):
        new_msg = {"role": "user", "content": f"give me the script for scene {i}"}
        response = openai.ChatCompletion.create(
          model="gpt-4",
          messages=input_messages + [new_msg]
        )
        system_msg = response.choices[0].message
        scripts.append(system_msg.content)

    while True:
        print(">> ")
        user_input = input()
        if user_input == "done":
            break
        print("\n")
        new_msg = {"role": "user", "content": user_input}
        input_messages.append(new_msg)
        response = openai.ChatCompletion.create(
          model="gpt-4",
          messages=input_messages
        )
        system_msg = response.choices[0].message

        print(f"GPT-4: {system_msg.content}")
        print("\n")
    scene_list = system_msg.content


def create_scene(text_input: str):
  print(f"Creating scene from text input: {text_input}")
  settings: List[Setting] = []
  scene_desc = ""
  shot = ""
  lines = ""
  for section in text_input.split("--"):
    section = section.strip("\n \t")
    if "Scene:" in section:
      scene_desc = section.split("Scene:")[1].strip()
    elif "Setting:" in section:
      cur_setting_shots: List[VideoShot | TalkingShot] = []
      cur_setting_desc = section.split("Setting:")[1].strip("\n \t")
      cur_setting = Setting(cur_setting_desc, cur_setting_shots)
      settings.append(cur_setting)
    elif "Line" in section:
      section_split = section.split("\n")
      line_unformatted = section_split[0].split("Line:")[1].strip("\n \t")
      speaker_split = line_unformatted.split(":")
      speaker = speaker_split[0]
      line = speaker_split[1].strip("\n \t")
      frame = section_split[1].split("Frame:")[1].strip("\n \t")
      cur_setting_shots.append(TalkingShot(shot=frame, lines=line, speaker=speaker))
    elif "Frame" in section:
      frame = section.split("Frame:")[1].strip("\n \t")
      cur_setting_shots.append(VideoShot(shot=frame))
  scene = Scene(scene_desc=scene_desc, settings=settings) 
  print(f"Created scene: {scene}")
  return scene


def create_scene_from_prompt(prompt: str):
    response = get_gpt4_response(prompt)
    return create_scene(response)

