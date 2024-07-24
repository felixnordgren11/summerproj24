import subprocess
import shutil
import os
import re
import numpy as np

output_folder = 'output_files'
original_file_path = 'inpsd.dat'
image_folders = ['layer_one', 'avg']



# Scaling values for the DM interaction strength
start = 0.74
step = 0.005
end = 4.0
amount = np.ceil((end - start) / step)
dm_scale_values = [round(start + step*i, 3) for i in range(1, int(amount)+1)]
print(len(dm_scale_values))

old_value = 0
old_fv = '0'


# Code for changing the J values, might make different files as to not have to change the original file.





#################################################################

#Ensure shell script is executable (that calls bin/sd.f95 to run the simulation)
if not os.access('./script_runner.sh', os.X_OK):
    os.chmod('./script_runner.sh', 0o755)
    
    


#################################################################
#################################################################




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
def move_and_clean(formatted_value):
    # Ensure the target directory exists
    for folder in image_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Define the file to save and move, as well as listing all the files to remove
    save_file = [f'mompix.dm_{formatted_value}.Z001.png', f'mompix.dm_{formatted_value}.png']
    files_to_remove = [
        f'mompix.dm_{formatted_value}.Z002.png',
        f'mompix.dm_{formatted_value}.Z003.png',
        f'mompix.dm_{formatted_value}.Z004.png',
        f'averages.dm_{formatted_value}.out',
        f'coord.dm_{formatted_value}.out',
        f'inp.dm_{formatted_value}.json',
        f'projavgs.dm_{formatted_value}.out',
        f'qm_minima.dm_{formatted_value}.out',
        f'qm_restart.dm_{formatted_value}.out',
        f'qm_sweep.dm_{formatted_value}.out',
        f'restart.dm_{formatted_value}.out',
        f'totenergy.dm_{formatted_value}.out',
        f'uppasd.dm_{formatted_value}.yaml',
        f'qpoints.out'
    ]
    
    # Move the specific file to the new directory
    for i in range(len(save_file)):
        if os.path.exists(save_file[i]):
            shutil.move(save_file[i], os.path.join(image_folders[i], save_file[i]))
            print(f"Moved {save_file[i]} to {image_folders[i]}")
        else:
            print(f"{save_file[i]} does not exist.")

    # Remove the unwanted files
    for file_name in files_to_remove:
        if os.path.exists(file_name):
            os.remove(file_name)
            #print(f"Removed {file_name}")
        else:
            print(f"{file_name} does not exist.")
            
            
            

#################################################################
#################################################################




# Reads the original input file. 
# Ensures it says 'simid dm_0' and 'dm_scale 0' when running 
# as to not need to change it when running a new simulation 
with open(original_file_path, 'r') as file:
    template_content = file.read()
    
current_simid = re.search(r'simid dm_(\d+)', template_content)
current_dm_scale = re.search(r'dm_scale ([\d\.]+)', template_content)

if current_simid:
    current_simid = current_simid.group(1)
else:
    current_simid = old_fv
    
if current_dm_scale:
    current_dm_scale = current_dm_scale.group(1)
else:
    current_dm_scale = old_value

# Check and update initial values if necessary
if current_dm_scale != str(old_value) or current_simid != old_fv:
    template_content = re.sub(r'dm_scale [\d\.]+', f'dm_scale {old_value}', template_content)
    template_content = re.sub(r'simid dm_\d+', f'simid dm_{old_fv}', template_content)

    with open(original_file_path, 'w') as file:
        file.write(template_content)




#################################################################
#################################################################




# Main loop. Replaces the dm_scale value and runs the image generator script
# then saves the desired image into layer(num)_images folder and removes the old 
# simulation result files that are not needed.

for value in dm_scale_values:

    # Format the value by removing the decimal point and having at least 4 digits.
    # Limits the value 'step size' to 0.0001.
    formatted_value = f"{abs(value):.4f}".replace('.', '')
    
    #for values above 0.9999 so that they have roughly the same format as below 1.
    formatted_value = formatted_value.zfill(4)
    
    
    #Make sure that the original inpsd.dat file has these lines when starting the script.
    new_content = template_content.replace(f"dm_scale {old_value}", f"dm_scale {value}")
    new_content = new_content.replace(f"simid dm_{old_fv}", f"simid dm_{formatted_value}")
    
    # Remove the original inpsd.dat file
    if os.path.exists(original_file_path):
        os.remove(original_file_path)
    
    #Write over the old file. Opens it for writing ('w')
    with open(original_file_path, 'w') as file:
        file.write(new_content)
    
    subprocess.run(['sh', './script_runner.sh'])
    
    if not run_script('python3', ['ASD_mompix.py']):
        print("Failed to run ASD_mompix.py, stopping execution.")
        break
    
    move_and_clean(formatted_value)
    
    template_content = new_content
    
    old_value = value
    old_fv = formatted_value
    
    
