import openai
import os.path
import dataclasses
from dataclasses import dataclass
from typing import List


PROMPT_TEMPLATE = """Write me a script for the Oppenheimer movie, but I need you to write it following my directions exactly. Every scene will begin with the prefix "Scene", and will have a brief description of what will happen. You must also describe each set that is used before you start giving shots that are taken there. Describe each shot in words so it can be used to generate an image with Stable Diffusion. The description of each shot needs to be rich, like you are painting a picture with words. If the shot has a person speaking in it, you must include the lines below the shot description. If a shot doesn't have a person speaking in it, then only have the prefix "Shot" with no lines section after it. Do not add any other sections, follow my directions exactly. Here's some examples, I need you to follow their formats exactly:
Scene: Oppenheimer starts his day.
--
Set: Oppenheimer's bedroom
--
Shot: Oppenheimer laying in bed under the covers, opening his eyes. 
Lines: Ahh, what a beautiful day..
--
Shot: Oppenheimer gets out of bed, stretching his arms. 
Lines: I think I smell breakfast cooking!
--
Set: Oppenheimer's kitchen
--
Shot: Wide shot of Oppenheimer entering his kitchen. His wife is there cooking.
--
Shot: Oppenheimer's wife cooking, looking over her shoulder at oppenheimer.
Lines: Morning, sweetie! Breakfast is almost ready, why dont you grab a seat?

That is the end of the example. Now I need you to write this scene: """


@dataclass
class VideoShot:
  shot: str

@dataclass
class TalkingShot:
  shot: str
  lines: str

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


def get_gpt4_response(prompt):
    with open(f"{os.path.dirname(__file__)}/../openai_key.txt", "r") as f:
        openai.api_key = f.read().strip()
    
    response = openai.Completion.create(
      engine="gpt-4",
      prompt=prompt,
    )

    return response.choices[0].text.strip()


def create_scene(text_input: str):
  settings: List[Setting] = []
  scene_desc = ""
  shot = ""
  lines = ""
  for section in text_input.split("--"):
    section = section.strip("\n")
    if "Scene:" in section:
      scene_desc = section.split("Scene:")[1].strip()
    elif "Set:" in section:
      cur_setting_shots: List[VideoShot | TalkingShot] = []
      cur_setting_desc = section.split("Set:")[1].strip()
      cur_setting = Setting(cur_setting_desc, cur_setting_shots)
      settings.append(cur_setting)
    elif "Shot" in section and "Lines" in section:
      section_split = section.split("\n")
      shot = section_split[0].split("Shot:")[1].strip()
      lines = section_split[1].split("Lines:")[1].strip()
      cur_setting_shots.append(TalkingShot(shot=shot, lines=lines))
    elif "Shot" in section:
      shot = section.split("Shot:")[1].strip()
      cur_setting_shots.append(VideoShot(shot=shot))
  return Scene(scene_desc=scene_desc, settings=settings)


def create_scene_from_prompt(prompt: str):
    response = get_gpt4_response(prompt)
    return create_scene(text_input)

