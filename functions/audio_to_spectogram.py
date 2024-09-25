#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 12:59:10 2023

@author: Henry
"""

import argparse
from pathlib import Path

import cv2
import librosa
import numpy as np
from tqdm import tqdm


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
    S = cv2.cvtColor(S, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(save_path, S)
