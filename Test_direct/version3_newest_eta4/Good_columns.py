import numpy as np
import sys


def process_file(input_file, output_file):
    # Read the data from the file
    data = np.loadtxt(input_file)

# Ensure data is 2D
    if data.ndim == 1:  # If only one row, reshape it
        data = data.reshape(1, -1)

    # List to hold the modified rows
    modified_data = data.copy()

    for i in range(0, len(modified_data)):  # Start from the first row
        # Access the values in the 3rd and 5th columns (index 2 and 4 in zero-indexed Python)
            val_3rd_col = modified_data[i][2]
            val_5th_col = modified_data[i][4]

            if abs(val_5th_col) < abs(val_3rd_col):
            # Swap the 2nd and 3rd columns with the 4th and 5th columns
                modified_data[i][1], modified_data[i][2], modified_data[i][3], modified_data[i][4], modified_data[i][5], modified_data[i][6], modified_data[i][7], modified_data[i][8] = modified_data[i][3], modified_data[i][4], modified_data[i][1], modified_data[i][2], modified_data[i][7], modified_data[i][8], modified_data[i][5], modified_data[i][6]
    np.savetxt(output_file, modified_data, fmt='%.10e')  # Save with scientific notation

#    print(f"File processed and saved to {output_file}")
# Example usage
#input_file = 'N2_minus.no.4states.cipsi.957210.scan2.data'  # Input file name
#output_file = 'processed_data.txt'  # Output file name
#process_file(input_file, output_file)


if __name__ == "__main__":
    # Check if the correct number of arguments are passed
    if len(sys.argv) != 3:
#        print("Usage: python script_name.py <input_file>")
        sys.exit(1)  # Exit if incorrect number of arguments

    # Get the input file path from command line argument
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Process the file
    process_file(input_file, output_file)
