import csv


def save_history(data, file_path):
    try:
        headers = list(data.keys())
        num_entries = len(next(iter(data.values())))  # Assuming all lists are of the same length

        # Check that all data lists are the same length
        if not all(len(lst) == num_entries for lst in data.values()):
            raise ValueError("All data lists must have the same number of entries.")

        # Write data to a CSV file
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)  # Write headers
            for i in range(num_entries):
                row = [data[header][i] for header in headers]  # Collect data for each header
                writer.writerow(row)  # Write data row
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")

