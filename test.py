# clean_requirements.py

import re

def is_local_path(line):
    # Matches lines with @ file://, @ /, or @ ~/
    return re.search(r'@\s*(file://|/|~/)', line) is not None

def clean_requirements(input_path, output_path=None):
    with open(input_path, 'r') as infile:
        lines = infile.readlines()

    cleaned_lines = [line for line in lines if not is_local_path(line)]

    if output_path is None:
        output_path = input_path  # Overwrite original

    with open(output_path, 'w') as outfile:
        outfile.writelines(cleaned_lines)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python clean_requirements.py requirements.txt [output.txt]")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        clean_requirements(input_file, output_file)
        print(f"Cleaned requirements written to {output_file or input_file}")