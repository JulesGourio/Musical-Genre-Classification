import os, sys
import pandas as pd


#This function labels the data in the filer and create a csv file with the tune name the extract number and the label

ex_label = "HardRock"

def labelize(filer_name, label):
    register_files = os.path.join(os.getcwd(), filer_name + label)
    for sound in os.listdir(register_files):
        sound_split = sound.split('_')
        df = pd.DataFrame({'tune': [sound_split[0]], 'extract': [sound_split[1]], 'label': [label]})
        df.to_csv('labels.csv', mode='a', header=False, index=False)        
    return df   

