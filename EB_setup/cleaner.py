import os, shutil, re, subprocess

# removes the unwanted files from the current directory before the new iteration.
def clean(formatted_value):
    
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
    
    # Remove the unwanted files
    for file_name in files_to_remove:
        if os.path.exists(file_name):
            os.remove(file_name)
            #print(f"Removed {file_name}")
        else:
            print(f"{file_name} does not exist.")
            
clean()