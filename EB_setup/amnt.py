import numpy as np

start = 0.4
step = 0.01
end = 1.2
amount = np.ceil((end - start) / step)
dm_scale_values = [round(start + step * i, 3) for i in range(1, int(amount) + 1)]

print(amount)