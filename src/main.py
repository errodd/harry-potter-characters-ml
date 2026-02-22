import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Cargar datasets
characters = pd.read_csv("data/Characters.csv", delimiter=';')
hp1 = pd.read_csv("data/Harry Potter 1.csv", delimiter=';')
hp2 = pd.read_csv("data/Harry Potter 2.csv", delimiter=';')
hp3 = pd.read_csv("data/Harry Potter 3.csv", delimiter=';')
spells = pd.read_csv("data/Spells.csv", delimiter=';')

# Renombrar columna de HP3 para que coincida
hp3 = hp3.rename(columns={"CHARACTER": "Character", "SENTENCE": "Sentence"})

hp1['Character'] = hp1['Character'].str.upper().str.strip()
hp2['Character'] = hp2['Character'].str.upper().str.strip()
hp3['Character'] = hp3['Character'].str.upper().str.strip()

hp2 = hp2[['Character', 'Sentence']]
hp3 = hp3[['Character', 'Sentence']]

hp_all = pd.concat([hp1, hp2, hp3], ignore_index=True)

# Contar líneas por personaje (sin repetir personajes)
lines_per_character = hp_all['Character'].value_counts()


spell_counts = []
for incantation in spells['Incantation'].dropna().unique():
    # Ignorar incantaciones vacías
    if incantation.strip() == '' or incantation.lower() == 'unknown':
        continue
    count = hp_all['Sentence'].str.contains(incantation, case=False, na=False).sum()
    spell_counts.append({'Incantation': incantation, 'TimesSaid': count})

# Crear DataFrame
spells_mentions_df = pd.DataFrame(spell_counts)

print(spells_mentions_df.head())