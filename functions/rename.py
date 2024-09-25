import os, sys

def rename_sounds(register_name):

    register_files = os.path.join(os.getcwd(), register_name)

    for sound in os.listdir(register_files):

        old_name = os.path.join(register_files, sound)
        new_name = sound[3:]
    
        os.rename(old_name, new_name)