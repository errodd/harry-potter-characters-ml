import pandas as pd

# ==============================
# CARGAR DATASETS (desde /data)
# ==============================

characters = pd.read_csv("../data/Characters.csv")
hp1 = pd.read_csv("../data/Harry Potter 1.csv")
hp2 = pd.read_csv("../data/Harry Potter 2.csv")
hp3 = pd.read_csv("../data/Harry Potter 3.csv")
spells = pd.read_csv("../data/Spells.csv")

# ==============================
# LIMPIEZA
# ==============================

# Eliminar duplicados
characters = characters.drop_duplicates()
spells = spells.drop_duplicates()

# ==============================
# MANEJO DE NULOS
# ==============================

characters = characters.fillna("Unknown")
spells = spells.fillna("Unknown")

# ==============================
# PROCESAMIENTO TEXTO (IMPORTANTE 🔥)
# ==============================

# Unir diálogos
hp_all = pd.concat([hp1, hp2, hp3])

# Limpiar nombres de columnas si hace falta
hp_all.columns = ["Character", "Sentence"]

# ==============================
# FEATURE ENGINEERING
# ==============================

# Contar líneas por personaje
lines_per_character = hp_all["Character"].value_counts().reset_index()
lines_per_character.columns = ["Character", "LineCount"]

# ==============================
# ENCODING
# ==============================

characters_selected = characters[[
    "Gender",
    "House",
    "Species",
    "Blood status"
]]

characters_encoded = pd.get_dummies(characters_selected)

# ==============================
# GUARDAR RESULTADOS
# ==============================

characters_encoded.to_csv("../data/characters_processed.csv", index=False)
lines_per_character.to_csv("../data/lines_per_character.csv", index=False)

print(" Data processing completado")