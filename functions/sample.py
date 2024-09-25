import os
import random
from pydub import AudioSegment


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
