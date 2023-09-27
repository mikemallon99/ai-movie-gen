import sys
from moviegen.storygen import get_gpt4_response


if __name__ == "__main__":
    # Check if there are enough arguments
    if len(sys.argv) != 3:
        print("Usage: python test_script_gen.py [input_file] [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, "r") as f:
        input_text = f.read().strip()

    response = get_gpt4_response(input_text)
    with open(output_file, "w") as f:
        f.write(response)
