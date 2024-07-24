import subprocess
import shutil
import os
import re
import numpy as np

output_folder = 'output_files'
original_file_path = 'inpsd.dat'
image_folders = ['layer_one', 'avg']

# Scaling values for the DM interaction strength
start = 0.4
step = 0.01
end = 1.2
amount = np.ceil((end - start) / step)
dm_scale_values = [round(start + step * i, 3) for i in range(1, int(amount) + 1)]

# Initialize old values for dm_scale and j0
old_dm_value = 0
old_dm_fv = '00000'
old_j0_value = '0.01'  # Assuming the initial j0 value is 0.01

filename = "jij"

# Generate replacement values for j0
replacement_values = [round(0.02 + 0.03 * (i - 1), 2) for i in range(1, 20)]

# Lines to edit (indexing from 0)
lines_to_edit = [40, 41, 42, 43]

#################################################################

# Ensure shell script is executable
if not os.access('./script_runner.sh', os.X_OK):
    os.chmod('./script_runner.sh', 0o755)

#################################################################
#################################################################

# Read the file
with open(filename, 'r') as file:
    data = file.readlines()

# Function to convert float to list of its string components
def float_to_strlist(number):
    str_number = f"{number:.2f}"
    result = [char for char in str_number]
    return result

# Function to check and replace the specific values in the line
def replace_value_in_line(line, old_value, new_value):
    pattern = re.compile(r'(\s*)(-?)' + re.escape(old_value) + r'(\s*)')
    new_value_str = ''.join(float_to_strlist(new_value))
    return pattern.sub(lambda match: match.group(1) + match.group(2) + new_value_str + match.group(3), line)

# Functions to help run scripts and manage files
def run_script(script, args):
    result = subprocess.run([script] + args, capture_output=True, text=True)
    print(f"Running {script} with arguments {args}")
    print(f"stdout: {result.stdout}")
    print(f"stderr: {result.stderr}")
    if result.returncode != 0:
        print(f"Error running {script}: {result.returncode}")
        return False
    return True

# Function that moves the desired image into new results folder and
# removes the unwanted files from the current directory before the new iteration.
def move_and_clean(formatted_value_dm, formatted_value_j0):
    # Ensure the target directory exists
    target_directory = os.path.join('layer_one', formatted_value_j0)
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Define the file to save and move, as well as listing all the files to remove
    save_file = [f'mompix.dm_{formatted_value_dm}.Z001.png']
    files_to_remove = [
        f'mompix.dm_{formatted_value_dm}.png',
        f'mompix.dm_{formatted_value_dm}.Z002.png',
        f'mompix.dm_{formatted_value_dm}.Z003.png',
        f'mompix.dm_{formatted_value_dm}.Z004.png',
        f'averages.dm_{formatted_value_dm}.out',
        f'coord.dm_{formatted_value_dm}.out',
        f'inp.dm_{formatted_value_dm}.json',
        f'projavgs.dm_{formatted_value_dm}.out',
        f'qm_minima.dm_{formatted_value_dm}.out',
        f'qm_restart.dm_{formatted_value_dm}.out',
        f'qm_sweep.dm_{formatted_value_dm}.out',
        f'restart.dm_{formatted_value_dm}.out',
        f'totenergy.dm_{formatted_value_dm}.out',
        f'uppasd.dm_{formatted_value_dm}.yaml',
        f'qpoints.out'
    ]

    # Move the specific file to the new directory
    for file in save_file:
        if os.path.exists(file):
            shutil.move(file, os.path.join(target_directory, file))
            print(f"Moved {file} to {target_directory}")
        else:
            print(f"{file} does not exist.")

    # Remove the unwanted files
    for file_name in files_to_remove:
        if os.path.exists(file_name):
            os.remove(file_name)
        else:
            print(f"{file_name} does not exist.")


#################################################################
#################################################################

# Reads the original input file.
# Ensures it says 'simid dm_00000_j0_000' and 'dm_scale 0' and 'j0 0' when running
# as to not need to change it when running a new simulation
with open(original_file_path, 'r') as file:
    template_content = file.read()

current_simid = re.search(r'simid dm_(\d+)_j0_(\d+)', template_content)
current_dm_scale = re.search(r'dm_scale ([\d\.]+)', template_content)

if current_simid:
    current_simid = current_simid.group(0)  # The entire matched string
else:
    current_simid = f'dm_{old_dm_fv.zfill(5)}'

if current_dm_scale:
    current_dm_scale = current_dm_scale.group(1)
else:
    current_dm_scale = str(old_dm_value)

# Check and update initial values if necessary
if current_dm_scale != str(old_dm_value) or current_simid != f'dm_{old_dm_fv.zfill(5)}':
    template_content = re.sub(r'dm_scale [\d\.]+', f'dm_scale {old_dm_value}', template_content)
    template_content = re.sub(r'simid dm_\d+_j0_\d+', f'simid dm_{old_dm_fv.zfill(5)}', template_content)

    with open(original_file_path, 'w') as file:
        file.write(template_content)

#################################################################
#################################################################

# Main loop. Replaces the dm_scale and j0 values and runs the image generator script
# then saves the desired image into layer(num)_images folder and removes the old
# simulation result files that are not needed.
for j0_value in replacement_values:
    # Replace j0 values in jij file
    for line_idx in lines_to_edit:
        if line_idx < len(data):
            line = data[line_idx]
            # Replace the old j0 value with the new j0 value
            data[line_idx] = replace_value_in_line(line, f'{float(old_j0_value):.2f}', j0_value)
    formatted_value_j0 = f"{j0_value:.2f}".replace('.', '').zfill(3)
    
    # Write the modified content back to the file
    with open(filename, 'w') as file:
        file.writelines(data)

    for dm_value in dm_scale_values:
        # Format the value by removing the decimal point and having at least 4 digits after decimal point.
        formatted_value_dm = f"{abs(dm_value):.4f}".replace('.', '').zfill(5)
        
        # Make sure that the original inpsd.dat file has these lines when starting the script.
        new_content = template_content.replace(f"dm_scale {old_dm_value}", f"dm_scale {dm_value}")
        new_content = re.sub(r'simid dm_\d+', f'simid dm_{formatted_value_dm}', new_content)

        # Write the new content to the input file
        with open(original_file_path, 'w') as file:
            file.write(new_content)

        # Run the simulation script
        if not run_script('./script_runner.sh', []):
            print("Failed to run script_runner.sh, stopping execution.")
            break
        
        if not run_script('python3', ['ASD_mompix.py']):
            print("Failed to run ASD_mompix.py, stopping execution.")
            break
        
        move_and_clean(formatted_value_dm, formatted_value_j0)
        
        # Update the old value for the next iteration
        template_content = new_content
        old_dm_value = dm_value
        old_dm_fv = formatted_value_dm
        old_j0_value = j0_value

        
    
    
