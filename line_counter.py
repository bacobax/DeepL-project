import os

def collect_python_files(root_dir):
    py_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".py") and not file.startswith("__init__"):
                full_path = os.path.join(dirpath, file)
                py_files.append(full_path)
    return sorted(py_files)

def count_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return len(f.readlines())

def main():
    root_dir = "./"  # change if needed
    files = collect_python_files(root_dir)

    total_lines = 0
    print(f"{'File':<60} | {'Lines'}")
    print("-" * 75)
    for file_path in files:
        line_count = count_lines(file_path)
        total_lines += line_count
        print(f"{file_path:<60} | {line_count}")

    print("-" * 75)
    print(f"{'TOTAL':<60} | {total_lines}")

if __name__ == "__main__":
    main()
