import random as R
import os

for s in [42, 43, 44, 45, 46]:
    R.seed(s)

    with open("VIsualAILab/codes/imageSets/train-all-02.txt", "r") as f:
        filenames = f.readlines()

    listLen = len(filenames)
    ps1 = int(listLen * 0.01) 
    ps5 = int(listLen * 0.05) 
    ps10 = int(listLen * 0.1) 
    ps50 = int(listLen * 0.5) 

    split_ps1 = R.sample(filenames, ps1)
    split_ps5 = R.sample(filenames, ps5)
    split_ps10 = R.sample(filenames, ps10)
    split_ps50 = R.sample(filenames, ps50)

    concat_dict = {"1":split_ps1, "5":split_ps5, 
                "10":split_ps10, "50":split_ps50}

    for id, split in concat_dict.items():
        with open(f"VIsualAILab/codes/imageSets/train-split-{id}ps_{s}seed.txt", "w") as f:
            for filename in split:
                f.write(filename)