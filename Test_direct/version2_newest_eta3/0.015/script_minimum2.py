import sys
import numpy as np
import matplotlib.pyplot as plt

def read_text_file(file_path):
    """Reads a file, handles empty or malformed lines, and converts data to a list of floats."""
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Iterate through each line
    for line in lines:
        line = line.strip()  # Remove leading/trailing spaces
        if len(line) > 0:  # Skip empty lines
            try:
                # Attempt to convert the line into a list of floats
                row = [float(i) for i in line.split()]
                data.append(row)  # Add the row to the data list
            except ValueError:
                # If there's an issue converting to float, skip the line
                print(f"Skipping malformed line: {line}")

    return data

def main():
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)

    # Get the file path from the command-line argument
    file_path = sys.argv[1]

    # Read the text file and create the NumPy array
    data = read_text_file(file_path)

    # Check if data is loaded
    if len(data) == 0:
        print("No valid data was loaded.")
        sys.exit(1)

    # Convert the list of lists to a NumPy array
    data = np.asarray(data, dtype=float)

    # Step 2: Split the data into x and y
    x = data[:, 0]  # First column: x values
#    y_columns = data[:, 1:]  # All columns after the first: y values

    # Example: Use the first column of y_columns for fitting (you can modify this if needed)
    y = data[:, -1]  # Assuming you want to use the first y column for fitting

    # Step 3: Fit the points to a 2nd-degree polynomial (quadratic)
    coefficients = np.polyfit(x, y, 2)

    # Print the coefficients (a, b, c for the polynomial ax^2 + bx + c)
#    print(f"Polynomial Coefficients: {coefficients}")

    # Step 4: Compute the derivative of the polynomial
    # The derivative of ax^2 + bx + c is 2ax + b
#    derivative_coeffs = np.polyder(coefficients)
#    print(f"Derivative Coefficients: {derivative_coeffs}")

    min_value = -coefficients[1]/(2*coefficients[0])

    # Step 5: Evaluate the derivative at x = 0
#    derivative_at_x0 = np.polyval(derivative_coeffs, 0)
#    print(f"{min_value:.5f}")

    y_pred = np.polyval(coefficients, x)
  # Compute the total sum of squares (SS_tot)
    ss_tot = np.sum((y - np.mean(y))**2)

    # Compute the residual sum of squares (SS_res)
    ss_res = np.sum((y - y_pred)**2)

    # R² value
    r2 = 1 - (ss_res / ss_tot)

    print(f"{min_value:.6f} {r2:.4f}")

    # Optional: Plot the data points and the fitted polynomial for visualization
#    x_fit = np.linspace(min(x), max(x), 100)
#    y_fit = np.polyval(coefficients, x_fit)
#
#    plt.figure(figsize=(8, 6))
#    plt.scatter(x, y, color='red', label='Data Points', alpha=0.6)
#    plt.plot(x_fit, y_fit, color='blue', label=f'Fitted Polynomial: {coefficients[0]:.2e}x² + {coefficients[1]:.2e}x + {coefficients[2]:.2e}')
#    plt.title('Fitted 2nd Degree Polynomial')
#    plt.xlabel('x')
#    plt.ylabel('y')
#    plt.legend()
#    plt.grid(True)
#    plt.show()

if __name__ == "__main__":
    main() 
