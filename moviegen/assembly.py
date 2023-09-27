import os.path
from moviegen.audiogen import make_tts_fakeyou, make_text_to_speech, SPONGEBOB_ID, PATRICK_ID
from moviegen.videogen import gen_talking_video, create_zeroscope_video, gen_still_video, make_stable_diffusion_image
from moviegen.storygen import Scene, create_scene_from_prompt, TalkingShot, VideoShot
from moviepy.editor import VideoFileClip, concatenate_videoclips


def concatenate_videos(input_files, output_file):
    clips = [VideoFileClip(clip) for clip in input_files]
    target_width, target_height = clips[0].size
    resized_clips = [clip.resize(newsize=(target_width, target_height)) for clip in clips]
    final_clip = concatenate_videoclips(resized_clips, method="compose")
    final_clip.write_videofile(output_file, codec="libx264", audio_codec="aac")


def create_video_from_scene(scene: Scene, out_file: str):
  video_paths = []
  print("generating clips from scenes")
  i = 0
  for setting in scene.settings:
    for shot in setting.shots:
      print(f"generating clip #{i}")
      clip_file = f'{os.path.dirname(__file__)}/../outputs/generated_video_{i}.mp4'

      image_prompt = shot.shot
      image_prompt += ", cinematic, ultrarealistic, 8k"
      # dialogue
      if isinstance(shot, TalkingShot):
        print(f"dialogue scene")
        audio_prompt = shot.lines
        print(f"Generating speech audio for text: {audio_prompt}")
        audio_path = f'{os.path.dirname(__file__)}/../outputs/generated_audio.wav'

        if shot.speaker.lower() == "spongebob":
          make_tts_fakeyou(SPONGEBOB_ID, audio_prompt, audio_path)
        elif shot.speaker.lower() == "patrick":
          make_tts_fakeyou(PATRICK_ID, audio_prompt, audio_path)
        else:
          make_text_to_speech(audio_prompt, audio_path)

        print(f"Generating image for prompt: {image_prompt}")
        image_path = f'{os.path.dirname(__file__)}/../outputs/generated_image.jpg'
        make_stable_diffusion_image(image_prompt, image_path)

        subtitle_text = f"{shot.speaker}: {shot.lines}"
        gen_talking_video(image_path, audio_path, subtitle_text, clip_file)
      # Still shot
      elif isinstance(shot, VideoShot):
        print("video shot")
        gen_still_video(image_prompt, clip_file)
      video_paths.append(clip_file)
      i += 1

  print("finished generating clips. concatenating to one video now.")
  concatenate_videos(video_paths, out_file)


def create_video_from_prompt(prompt: str, out_file: str):
    scene = create_scene_from_prompt(prompt)
    create_video_from_scene(scene, out_file)

