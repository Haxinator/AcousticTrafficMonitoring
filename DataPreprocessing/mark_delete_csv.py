import os
import csv
import shutil

# Assume script is one level above deleted directory and csv file.
script_dir = os.path.dirname(os.path.abspath(__file__))
deleted_dir = os.path.join(script_dir, 'vehicle_segments_peak1/deleted')
csv_path = os.path.join(script_dir, 'vehicle_segments_peak1/vehicle_clips.csv')

try:
    deleted_files = set(os.listdir(deleted_dir))
    print(f"Found {len(deleted_files)} files in the 'deleted' directory.")
except FileNotFoundError:
    print(
        f"Error: The directory '{deleted_dir}' was not found. Please check your path.")
    deleted_files = set()

temp_csv_path = csv_path + '.tmp'
updated_rows = []
marked_count = 0

try:
    with open(csv_path, mode='r', newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader)

        if 'deleted' not in header:
            header.append('deleted')

        filepath_idx = header.index('filepath')
        deleted_idx = header.index('deleted')

        updated_rows.append(header)

        for row in reader:
            if len(row) > filepath_idx:
                filepath = os.path.basename(row[filepath_idx])
                filename = filepath.split('\\')[1]

                if filename in deleted_files:
                    if len(row) < deleted_idx + 1:
                        row.extend([''] * (deleted_idx + 1 - len(row)))

                    row[deleted_idx] = 'x'
                    marked_count += 1

            updated_rows.append(row)

    print(f"{marked_count} files marked with x")
    with open(temp_csv_path, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(updated_rows)

    shutil.move(temp_csv_path, csv_path)
    print(
        f"\nProcessing complete. The CSV file has been updated in place: {csv_path}")

except FileNotFoundError as e:
    print(f"Error: A file or directory was not found. Details: {e}")
except ValueError as e:
    print(f"Error: A column was not found. Details: {e}")
    print("Please ensure your CSV has a column named 'filepath' and the data is correct.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
