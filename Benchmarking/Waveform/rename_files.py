import glob
import os

all_files = glob.glob('FedClem_WF_*.mat')
count = 1
for file in all_files:
    new_name = 'FedClem_WF_' + str(count) + '.mat'
    os.rename(file, new_name)
    count += 1