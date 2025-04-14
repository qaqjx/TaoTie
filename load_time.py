import re

# Function to extract and sum all load times from the file
def sum_load_times(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract all load times using regular expression
    load_times = re.findall(r'divide time:\s+(\d+\.\d+)', content)
    load_times = [float(t) for t in load_times]
    
    return sum(load_times), len(load_times)

# Calculate the sum of all load times
total_load_time, count = sum_load_times('/home/xujie/TaoTie/txt.txt')

print(f"Total load time: {total_load_time:.6f} seconds")
print(f"Number of load time entries: {count}")
print(f"Average load time: {total_load_time/count:.6f} seconds")