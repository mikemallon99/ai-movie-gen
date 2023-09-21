from diffusers import DiffusionPipeline
import torch

def get_stable_diffusion_model():
    stable_diffusion_model = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    stable_diffusion_model.to("cuda")

    return stable_diffusion_model


def make_stable_diffusion_image(prompt, out_file):
  model = get_stable_diffusion_model()
  images = model(prompt=prompt).images[0]
  images.save(out_file)


import soundfile as sf
import numpy as np
from transformers import AutoProcessor, AutoModel

def get_audio_models():
  audio_processor = AutoProcessor.from_pretrained("suno/bark-small")
  audio_model = AutoModel.from_pretrained("suno/bark-small").to("cuda")
  return audio_processor, audio_model


def make_text_to_speech(prompt, out_file):
  processor, model = get_audio_models()

  inputs = processor(
      text=[prompt],
      return_tensors="pt",
      voice_preset="v2/en_speaker_6"
  )

  speech_values = model.generate(**inputs.to("cuda"), do_sample=True)

  # Convert the tensor to a numpy array
  audio_np = speech_values.squeeze().cpu().numpy()

  # Normalize the audio to be between -1 and 1 (if it's not already)
  audio_np = np.interp(audio_np, (audio_np.min(), audio_np.max()), (-1, 1))

  # Save the audio to a wav file
  sf.write(out_file, audio_np, samplerate=22050)  # Assuming a sample rate of 22,050Hz, you might need to adjust this.

  print(f"Audio saved to {out_file}")


import subprocess

def run_wav2lip(image_path, audio_path, out_file):
    command = [
        'python3',
        'wav2lip_model/inference.py',
        '--checkpoint_path', 'wav2lip_model/checkpoints/wav2lip_gan.pth',
        '--face', image_path,
        '--audio', audio_path,
        '--outfile', out_file
    ]

    print(f"Running wav2lip function, output to path {out_file}")
    subprocess.run(command)


def create_ai_video(image_prompt, audio_prompt, out_file):
  print(f"Generating image for prompt: {image_prompt}")
  image_path = 'outputs/generated_image.jpg'
  make_stable_diffusion_image(image_prompt, image_path)

  print(f"Generating speech audio for text: {audio_prompt}")
  audio_path = 'outputs/generated_audio.wav'
  make_text_to_speech(audio_prompt, audio_path)

  print(f"Generating lip sync video...")
  run_wav2lip(
      image_path=image_path,
      audio_path=audio_path,
      out_file=out_file
  )


from moviepy.editor import VideoFileClip, concatenate_videoclips
import sys

def concatenate_videos(input_files, output_file):
    clips = [VideoFileClip(video) for video in input_files]
    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(output_file, codec="libx264", audio_codec="aac")


from moviepy.editor import ImageClip

def generate_still_video(image_prompt, out_file):
  print(f"Generating image for prompt: {image_prompt}")
  image_path = 'outputs/generated_image.jpg'
  make_stable_diffusion_image(image_prompt, image_path)

  # Convert the image into a 3-second clip
  clip = ImageClip(image_path)
  clip = clip.set_duration(3)  # Set the duration for the clip to 3 seconds

  # Write the clip to an mp4 file
  print(f"Writing video to path {out_file}")
  clip.write_videofile(out_file, fps=24)


import dataclasses
from dataclasses import dataclass
from typing import List

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


def create_video_from_scene(scene: Scene, out_file: str):
  video_paths = []
  print("generating clips from scenes")
  i = 0
  for setting in scene.settings:
    for shot in setting.shots:
      print(f"generating clip #{i}")
      clip_file = f'outputs/generated_video_{i}.mp4'
      image_prompt = f"{shot.shot} Taken at {setting.set_desc}"
      # dialogue
      if isinstance(shot, TalkingShot):
        print(f"dialogue scene")
        audio_prompt = shot.lines
        create_ai_video(image_prompt, audio_prompt, clip_file)
      # Still shot
      elif isinstance(shot, VideoShot):
        print("still shot")
        generate_still_video(image_prompt, clip_file)
      video_paths.append(clip_file)
      i += 1

  print("finished generating clips. concatenating to one video now.")
  concatenate_videos(video_paths, out_file)

second_try_text = """Scene: Oppenheimer's Workday Commencement
--
Set: Oppenheimer's Home Exterior and Street
--
Shot: The front of a modest two-story home, with Oppenheimer stepping out of the front door, briefcase in hand, wearing a crisp suit. The morning sun casts a warm light, and the chirping of birds can be heard.
--
Shot: Oppenheimer locking his front door and then taking a moment to adjust his hat and straighten his tie.
Lines: Let's make history today.
--
Shot: A side shot of Oppenheimer walking down the sidewalk, neighbors waving at him as they tend to their gardens or collect their morning newspapers.
--
Set: Los Alamos Laboratory Entrance
--
Shot: The exterior of the laboratory, a blend of rustic and industrial architecture. Armed guards stand at the entrance, checking identification of those entering. The sign overhead reads "Los Alamos Laboratory."
--
Shot: Oppenheimer approaching the entrance, nodding at one of the guards.
--
Shot: Close-up of Oppenheimer showing his identification to the guard.
--
Shot: The guard saluting Oppenheimer respectfully.
Lines: Good morning, Dr. Oppenheimer.
--
Shot: Oppenheimer walking through the entrance, the weight of his responsibilities evident in his posture.
Lines: Morning. Let's see where we stand today.
--
Set: Oppenheimer's Office at Los Alamos
--
Shot: A spacious office with wooden bookshelves lining the walls filled with books, papers, and scientific equipment. A large wooden desk dominates the room with a chair behind it. There are papers, schematics, and a few pictures on the desk. The room has an aura of intellectual rigor.
--
Shot: Oppenheimer entering his office, placing his briefcase on his desk and sitting down.
--
Shot: Oppenheimer taking a deep breath, looking at a picture of his family on his desk.
Lines: For them... for all of them.
--
Shot: Oppenheimer beginning to sift through the papers on his desk, preparing for the day's work.
Lines: Let's get to it.
--
"""


out_file = 'outputs/generated_full_movie_4.mp4'
scene = create_scene(second_try_text)
create_video_from_scene(scene, out_file)
display_video(out_file)
