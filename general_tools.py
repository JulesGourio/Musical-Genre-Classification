#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thursday September 27 2024

@author: Maxime
"""

import os, sys

import argparse
from pathlib import Path
import shutil

import cv2
import librosa
import numpy as np
from tqdm import tqdm

import random
from pydub import AudioSegment


def list_files(source):
    """
    List all files in the given source directory and its subdirectories.

    Args:
        source (str): The source directory path.

    Returns:
        list: A list of `Path` objects representing the files.
    """
    path = Path(source)
    files = [file for file in path.rglob('*') if file.is_file()]
    return files


def audio_to_spectrogram(audio_path, save_path, duration):
    """
    Convert an audio file to a spectrogram and save it as an image.

    Args:
        audio_path (str): The path to the audio file.
        save_path (str): The path to save the spectrogram image.
        duration (int): Duration of the audio file to process in seconds.

    Returns:
        None
    """
    # Load audio file
    y, sr = librosa.load(audio_path, duration=duration)

    # Compute spectrogram
    D = librosa.stft(y)
    S = librosa.amplitude_to_db(abs(D), ref=np.max)

    # Normalize values to 0-255 range and convert to uint8
    S = cv2.normalize(S, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Convert to RGB and save as PNG
    #S = cv2.cvtColor(S, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(save_path, S)



def rename_sounds(register_name):

    register_files = os.path.join(os.getcwd(), register_name)

    for sound in os.listdir(register_files):

        old_name = os.path.join(register_files, sound)
        new_name = os.path.join(register_files, sound[3:])
    
        os.rename(old_name, new_name)



# Fonction pour extraire une partie aléatoire de 30s d'un fichier .mp3
def extrait_aleatoire(fichier_mp3, duree_extrait):
    # Charger le fichier audio
    audio = AudioSegment.from_mp3(fichier_mp3)
    
    # Calculer la durée totale de l'audio en millisecondes
    duree_totale = len(audio)
    
    # Si la durée de l'audio est inférieure à 30s, on prend tout l'audio
    if duree_totale <= duree_extrait:
        return audio
    
    # Choisir un point de départ aléatoire
    debut_extrait = random.randint(0, duree_totale - duree_extrait)
    
    # Extraire la partie de 30 secondes
    extrait = audio[debut_extrait:debut_extrait + duree_extrait]
    
    return extrait



def copy_paste(source_path, destination_path):

    shutil.copy(source_path, destination_path)


def custom_sort(columns):
    return columns.str[10:]

