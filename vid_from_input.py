import sys
from moviegen.assembly import create_video_from_scene
from moviegen.storygen import create_scene


if __name__ == "__main__":
    # Check if there are enough arguments
    if len(sys.argv) != 3:
        print("Usage: python vid_from_input.py [input_file] [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_filename = sys.argv[2]

    with open(input_file, "r") as f:
        input_text = f.read().strip()
        scene = create_scene(input_text)

    create_video_from_scene(scene, output_filename)

