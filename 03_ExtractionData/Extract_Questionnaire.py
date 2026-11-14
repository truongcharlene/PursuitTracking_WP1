"""
Gather the questionnaires completed after training (session 3) for all
participants into a single Excel file.
-------------
For each *Session_3*.xlsx file found in DATA_FOLDER:
  - sheet "QV"   : questionnaire responses
  - sheet "NASA" : NASA-TLX (perceived workload)
Each row is tagged with the source file name (column "Fichier").

Output: Recapitulatif_Session_3.xlsx (saved in the current working directory),
        with one sheet "QV" and one sheet "NASA".
-------------
Author: Charlène Truong - Brain, Action and Skill Lab, UCLouvain
"""

import pandas as pd
import os
import glob

# Folder containing the session 3 files (.xlsx), searched recursively -> to adapt
folder_path = r"path/to/All_Data"

qv_results = []
nasa_results = []

# Browse all Session_3 files
for file in glob.glob(os.path.join(folder_path, "**", "*Session_3*.xlsx"), recursive=True):

    file_name = os.path.basename(file)
    print(file)
    xls = pd.ExcelFile(file)

    # "QV" sheet
    if "QV" in xls.sheet_names:
        df_qv = pd.read_excel(file, sheet_name="QV")
        df_qv["Fichier"] = file_name
        qv_results.append(df_qv)

    # "NASA" sheet (NASA-TLX)
    if "NASA" in xls.sheet_names:
        df_nasa = pd.read_excel(file, sheet_name="NASA")
        df_nasa["Fichier"] = file_name
        nasa_results.append(df_nasa)

# Concatenate all participants
qv_final = pd.concat(qv_results, ignore_index=True)
nasa_final = pd.concat(nasa_results, ignore_index=True)

# Write both tables in a single Excel file
# (saved in the current working directory, like the other scripts)
output_file = "Questionnaires_data.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    # header=True: column names written in the first row
    qv_final.to_excel(writer, sheet_name="QV", index=False, header=True)
    nasa_final.to_excel(writer, sheet_name="NASA", index=False, header=True)

print(f"File created: {output_file}")