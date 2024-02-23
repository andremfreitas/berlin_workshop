import pandas as pd
import numpy as np

# Replace 'your_file.csv' with the actual file path
file_path = 'camelyonpatch_level_2_split_test_meta.csv'

# Load the CSV data into a Pandas DataFrame
df = pd.read_csv(file_path, sep=',')

center_tumor_patch_data = df.iloc[:, 3]
wsi_data = df.iloc[:, 4]

data1 = np.array(center_tumor_patch_data)
data2 = np.array(wsi_data)

# test = np.array_equal(data1, data2)
data1 = data1.astype(int)

# counter = 0
# for i in range(data1.shape[0]):
#     if data1[i] == data2[i]:
#         pass
#     else:
#         counter += 1



        


