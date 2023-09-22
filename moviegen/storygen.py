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


STORYBOARD_SYSTEM_PROMPT = """You are ScriptGPT, an ai bot that creates entire scripts out of outlines. I need you to follow my directions exactly. Do not add any extra formatting or responses. I will give you my input as sections separated with a "--". There is an "Act" section, which gives a description of the act. Then, there will be a "Scene" section for each scene. I'd like you to expand on each scene by giving a full script for each scene, with sections "Setting", "Action", "Line", and if you have a "Line" section then you must start it by giving the character who's saying the line, like this "Line: Oppenheimer: <line>". Additionally, I need a "Frame" section paired with each "Action" and "Line" section which describes a storyboard version of the image. Please provide as much detail of the image as you could, as if you were giving instructions to a painter making the storyboard frame. Give details like how the shot is framed, the environment, all the characters in the frame. Also, provide the full context of the image as there needs to be continuity across each story board frame. Here is an example of what I'm looking for:
Act: The Manhattan Project
--
Scene: Oppenheimer's recruitment into the secret project, highlighting the urgency and stakes.
--
Setting: A dimly lit room in a secret government facility, filled with documents, classified blueprints, and a few officials seated around a table.
--
Action: The door opens, and a man, General Groves, gestures for Oppenheimer to come in.
Frame: The door slowly creaks open revealing a backlighted silhouette of General Groves. The framing is a medium shot with the General centered. The dimly lit room with officials, blueprints, and papers are blurry in the background.
--
Line: General Groves: Oppenheimer, glad you could make it.
Frame: Close-up on General Groves. His face is stern but relieved. The dim light casts shadows on his face, emphasizing his wrinkles and determination.
--
Action: Oppenheimer nods and takes a seat.
Frame: Oppenheimer standing near the door, silhouetted by the light behind him, nodding his head. Medium shot framing, shot from under him and slightly to the side, capturing the table with officials in the foreground.
--
Line: Oppenheimer: General. I’ve heard whispers. What's so urgent?
Frame: Oppenheimer sitting down with the officials surrounding the table. Medium close-up. He's speaking with a mix of curiosity and apprehension, eyes locked on General Groves.
--
Line: Official 1: Dr. Oppenheimer, this is a matter of utmost national security.
Frame: Close-up on Official 1. He looks older, with grey hair, stern, wearing spectacles. His voice carries gravity, emphasizing the importance of his words.
--
Action: General Groves pushes a folder of documents toward Oppenheimer.
Frame: A mid-shot showing General Groves' hand pushing a manila folder across the table. Oppenheimer's hands are in the frame's bottom, showing anticipation as the folder moves closer to him.
"""


STORYBOARD_USER_PROMPT = """Please provide me a large script for this input:
--
Act: The Manhattan Project
--
Scene: Oppenheimer's recruitment into the secret project, highlighting the urgency and stakes.
--
Scene: Collaboration and conflicts among scientists, including tensions with Edward Teller.
--
Scene: The trials and tribulations of designing and testing the bomb, emphasizing Oppenheimer's evolving moral dilemma.
--
Scene: The successful test at Trinity site, juxtaposed with Oppenheimer's famous line, "Now I am become Death, the destroyer of worlds."
"""


SYSTEM_PROMPT = """You are ScriptGPT, an AI bot which generates movie scripts that follow a very specific format. Every scene will begin with the prefix "Scene", and will have a brief description of what will happen. You must also describe each set that is used before you start giving shots that are taken there. Describe each shot in words so it can be used to generate an image with Stable Diffusion. The description of each shot needs to be rich, like you are painting a picture with words. If the shot has a person speaking in it, you must include the lines below the shot description. If a shot doesn't have a person speaking in it, then only have the prefix "Shot" with no lines section after it. Do not add any other sections, follow my directions exactly. Here's some examples, you need to follow their formats exactly:
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
--"""

PROMPT_TEMPLATE = """I need you to write this scene: """


@dataclass
class VideoShot:
  shot: str
  action: str

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
    print(f"Calling GPT4 with prompt: {prompt}")
    with open(f"{os.path.dirname(__file__)}/../openai_key.txt", "r") as f:
        openai.api_key = f.read().strip()
    
    system_prompt = STORYBOARD_SYSTEM_PROMPT
    user_prompt = STORYBOARD_USER_PROMPT
    print(f"Calling storyboard GPT4 with user prompt: {user_prompt}")
    response = openai.ChatCompletion.create(
      model="gpt-4",
      messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
      ]
    )
    output = response.choices[0].message.content.strip()
    print(f"Output from GPT: {output}")
    return output

def create_scene(text_input: str):
  print(f"Creating scene from text input: {text_input}")
  settings: List[Setting] = []
  scene_desc = ""
  shot = ""
  lines = ""
  for section in text_input.split("--"):
    section = section.strip("\n")
    if "Scene:" in section:
      scene_desc = section.split("Scene:")[1].strip()
    elif "Setting:" in section:
      cur_setting_shots: List[VideoShot | TalkingShot] = []
      cur_setting_desc = section.split("Setting:")[1].strip()
      cur_setting = Setting(cur_setting_desc, cur_setting_shots)
      settings.append(cur_setting)
    elif "Line" in section:
      section_split = section.split("\n")
      lines = section_split[0].split("Line:")[1].strip()
      frame = section_split[1].split("Frame:")[1].strip()
      cur_setting_shots.append(TalkingShot(shot=frame, lines=lines))
    elif "Action" in section:
      section_split = section.split("\n")
      action = section_split[0].split("Action:")[1].strip()
      frame = section_split[1].split("Frame:")[1].strip()
      cur_setting_shots.append(VideoShot(shot=frame, action=action))
  scene = Scene(scene_desc=scene_desc, settings=settings) 
  print(f"Created scene: {scene}")
  return scene


def create_scene_from_prompt(prompt: str):
    response = get_gpt4_response(prompt)
    return create_scene(response)

