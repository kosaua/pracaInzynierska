import os

# Folder z plikami
SOURCE_DIR = "."  # aktualny folder

OUTPUT_PY = "wszystkie_pliki_py.txt"
OUTPUT_SH = "wszystkie_pliki_sh.txt"


def zapisz_pliki(extension, output_file):
    with open(output_file, "w", encoding="utf-8") as outfile:
        for file in os.listdir(SOURCE_DIR):
            file_path = os.path.join(SOURCE_DIR, file)

            # Tylko pliki z odpowiednim rozszerzeniem
            if os.path.isfile(file_path) and file.endswith(extension):

                # Nagłówek z nazwą pliku
                outfile.write(f"\n{'=' * 80}\n")
                outfile.write(f"PLIK: {file_path}\n")
                outfile.write(f"{'=' * 80}\n\n")

                try:
                    with open(file_path, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())
                        outfile.write("\n\n")
                except Exception as e:
                    outfile.write(f"[BŁĄD ODCZYTU: {e}]\n\n")


# Zapis plików .py
zapisz_pliki(".py", OUTPUT_PY)

# Zapis plików .sh
zapisz_pliki(".sh", OUTPUT_SH)

print(f"Gotowe.")
print(f"Pliki .py zapisano do: {OUTPUT_PY}")
print(f"Pliki .sh zapisano do: {OUTPUT_SH}")