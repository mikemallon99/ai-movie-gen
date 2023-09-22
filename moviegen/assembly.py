import os.path
from moviegen.videogen import gen_talking_video, gen_still_video
from moviegen.storygen import Scene, create_scene_from_prompt, TalkingShot, VideoShot
from moviepy.editor import VideoFileClip, concatenate_videoclips


def concatenate_videos(input_files, output_file):
    clips = [VideoFileClip(video) for video in input_files]
    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(output_file, codec="libx264", audio_codec="aac")


def create_video_from_scene(scene: Scene, out_file: str):
  video_paths = []
  print("generating clips from scenes")
  i = 0
  for setting in scene.settings:
    for shot in setting.shots:
      print(f"generating clip #{i}")
      clip_file = f'{os.path.dirname(__file__)}/outputs/generated_video_{i}.mp4'
      image_prompt = f"{shot.shot} Photo taken at {setting.set_desc}"
      # dialogue
      if isinstance(shot, TalkingShot):
        print(f"dialogue scene")
        audio_prompt = shot.lines
        gen_talking_video(image_prompt, audio_prompt, clip_file)
      # Still shot
      elif isinstance(shot, VideoShot):
        print("still shot")
        gen_still_video(image_prompt, clip_file)
      video_paths.append(clip_file)
      i += 1

  print("finished generating clips. concatenating to one video now.")
  concatenate_videos(video_paths, out_file)


def create_video_from_prompt(prompt: str, out_file: str):
    scene = create_scene_from_prompt(prompt)
    create_video_from_scene(scene, out_file)

