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
