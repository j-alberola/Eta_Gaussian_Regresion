import sys

def find_minimum_and_surrounding_rows(filename, column_index=7):
    # Open the file and read the data
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Strip newline characters
    lines = [line.strip() for line in lines]

    # Find the minimum value and its corresponding line number
    min_value = float('inf')
    min_line_idx = -1

    # Iterate through the lines and find the minimum in the specified column
    for idx, line in enumerate(lines):
        try:
            value = float(line.split()[column_index])  # assuming space-separated columns
            if value < min_value:
                min_value = value
                min_line_idx = idx
        except ValueError:
            continue  # Skip lines that can't be converted to a float

    # Now print the surrounding rows (2 before, 2 after, with edge case handling)
#    print(f"Minimum value: {min_value} found in line {min_line_idx + 1}")

    # Print the rows before and after the minimum
    start_idx = max(0, min_line_idx - 5)  # Two rows before
    end_idx = min(len(lines), min_line_idx + 5)  # Two rows after

    for i in range(start_idx, end_idx):
        print(lines[i])

if __name__ == "__main__":
    # Get the filename from command-line arguments
    if len(sys.argv) != 2:
        print("Usage: script_minimum.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    find_minimum_and_surrounding_rows(filename)
