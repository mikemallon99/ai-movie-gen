import sys
from moviegen.assembly import create_video_from_prompt


if __name__ == "__main__":
    # Check if there are enough arguments
    if len(sys.argv) != 3:
        print("Usage: python new_video.py [prompt] [output_file]")
        sys.exit(1)

    prompt_text = sys.argv[1]
    output_filename = sys.argv[2]

    create_video_from_prompt(prompt_text, output_filename)

