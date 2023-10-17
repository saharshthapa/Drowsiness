import csv
from datetime import datetime
import pandas as pd

# Read the CSV file

data=pd._csv("./eye_logs/eye_log11.csv" )
# Initialize variables
blink_count = 0
before_state = 1
time_at_before = None

with open(csv_file_path, "r") as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  # Skip the header row

    for row in csv_reader:
        timestamp, right_eye_state, left_eye_state = row

        present_state = 1 

        if before_state == 0 and present_state == 1:
            time_at_before = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

        if before_state == 1 and present_state == 0:
            blink_count += 1
            if time_at_before is not None:
                current_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                duration = (current_time - time_at_before).total_seconds()
                if duration < 1:
                    duration = 0.3
                print(f"Blink {blink_count}: Duration = {duration:.2f} seconds")

        before_state = present_state

print(f"Total blinks: {blink_count}")
