import re

filename = "jij_exp"

# Generate replacement values up to 1.99 with two decimal places
replacement_values = [round(0.01 * i, 2) for i in range(1, 20)]

# Lines to edit (indexing from 0)
lines_to_edit = [40, 41, 42, 43]

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

# Iterate over the lines to edit and replace the values
for i in range(1, len(replacement_values)):
    for line_idx in lines_to_edit:
        if line_idx < len(data):
            line = data[line_idx]
            # Replace the old replacement value with the new replacement value
            data[line_idx] = replace_value_in_line(line, f'{replacement_values[i-1]:.2f}', replacement_values[i])

    # Write the modified content back to the file
    with open(filename, 'w') as file:
        file.writelines(data)

    # Print the modified lines for verification
    for line_idx in lines_to_edit:
        if line_idx < len(data):
            print(data[line_idx], end='')
    print(' ')

