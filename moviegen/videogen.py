import os
import os.path
import torch
import subprocess
from diffusers import DiffusionPipeline
from moviepy.editor import ImageClip, AudioFileClip
from moviegen.audiogen import make_text_to_speech


def get_stable_diffusion_model():
    stable_diffusion_model = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    stable_diffusion_model.to("cuda")

    return stable_diffusion_model


def make_stable_diffusion_image(prompt, out_file):
  model = get_stable_diffusion_model()
  images = model(prompt=prompt).images[0]
  images.save(out_file)


def combine_image_and_audio(image_path, audio_path, out_file):
    print(f"Attempting to combine image: {image_path} ; and audio {audio_path}")
    img_clip = ImageClip(image_path)
    audio = AudioFileClip(audio_path)
    img_clip = img_clip.set_duration(audio.duration)
    video = img_clip.set_audio(audio)
    video.write_videofile(out_file, fps=24)
    print(f"Successfully combined image and audio to path {out_file}")


def run_wav2lip(image_path, audio_path, out_file):
    prev_dir = os.getcwd()
    os.chdir(f"{os.path.dirname(__file__)}/../models/wav2lip_model")
    command = [
        'python3',
        'inference.py',
        '--checkpoint_path', f"{os.path.dirname(__file__)}/../models/wav2lip_model/checkpoints/wav2lip_gan.pth",
        '--face', image_path,
        '--audio', audio_path,
        '--outfile', out_file
    ]

    print(f"Running wav2lip function, output to path {out_file}")
    result = subprocess.run(command)
    os.chdir(prev_dir)
    return result.returncode


def gen_talking_video(image_prompt, audio_prompt, out_file):
  print(f"Generating image for prompt: {image_prompt}")
  image_path = f'{os.path.dirname(__file__)}/../outputs/generated_image.jpg'
  make_stable_diffusion_image(image_prompt, image_path)

  print(f"Generating speech audio for text: {audio_prompt}")
  audio_path = f'{os.path.dirname(__file__)}/../outputs/generated_audio.wav'
  make_text_to_speech(audio_prompt, audio_path)
  combine_image_and_audio(image_path, audio_path, out_file)
  return

  print(f"Generating lip sync video...")
  return_code = run_wav2lip(
      image_path=image_path,
      audio_path=audio_path,
      out_file=out_file
  )
  if return_code != 0:
    print("Wav2Lip failed. Creating a video without lip sync...")
    combine_image_and_audio(image_path, audio_path, out_file)


def gen_still_video(image_prompt, out_file):
  print(f"Generating image for prompt: {image_prompt}")
  image_path = f"{os.path.dirname(__file__)}/../outputs/generated_image.jpg"
  make_stable_diffusion_image(image_prompt, image_path)

  # Convert the image into a 3-second clip
  clip = ImageClip(image_path)
  clip = clip.set_duration(3)  # Set the duration for the clip to 3 seconds

  # Write the clip to an mp4 file
  print(f"Writing video to path {out_file}")
  clip.write_videofile(out_file, fps=24)

