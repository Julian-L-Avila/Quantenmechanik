#!/bin/bash

# Definiere das Verzeichnis, in dem sich die .txt-Dateien befinden
directory="./"  # Aktuelles Verzeichnis als Standard

# Schleife durch alle .txt-Dateien im angegebenen Verzeichnis
find "$directory" -maxdepth 1 -type f -name "*.txt" -print0 | while IFS= read -r -d $'\0' file; do
  # Erstelle den Namen für die neue .tvs-Datei
  new_file="${file%.txt}.tsv"

  # Bearbeite die ersten 5 Zeilen und schreibe in die neue Datei
  head -n 5 "$file" | sed 's/^/#/' > "$new_file"
  tail -n +6 "$file" >> "$new_file"

  echo "Datei '$file' wurde zu '$new_file' transformiert."
done

echo "Transformation abgeschlossen."
