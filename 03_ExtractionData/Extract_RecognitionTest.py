"""
Extract the recognition test answers for each participant.
-------------
For each *Recognitiontest.xlsx file found in DATA_FOLDER,
and for each sheet, the script reads the 4 first columns (A to D) and retrieves
the response (column D) for the three types of sequence presented:
  - Rsequence : repeated sequence  (C == A and B == 'no')
  - Rmirror   : mirror sequence    (C == A and B == 'yes')
  - Ralea     : random sequence    (C != A and B == 'no')

Output: Recognition_data.xlsx (one row per file x sheet)
-------------
Author: Charlène Truong - Brain, Action and Skill Lab, UCLouvain
"""

import pandas as pd
import os
import glob

# Folder containing the recognition files (.xlsx), searched recursively -> to adapt
folder_path = r"path/to/All_Data"

results = []

# Browse all recognition test files
for file in glob.glob(os.path.join(folder_path, "**", "*Recognitiontest.xlsx"), recursive=True):

    file_name = os.path.basename(file)

    # Read every sheet of the file
    xls = pd.ExcelFile(file)

    for sheet in xls.sheet_names:
        df = pd.read_excel(file, sheet_name=sheet)

        # Columns A to D (index 0 to 3)
        col_A = df.iloc[:, 0]
        col_B = df.iloc[:, 1]
        col_C = df.iloc[:, 2]
        col_D = df.iloc[:, 3]   # participant's response

        # Response for each sequence type (first matching row)
        sequence = col_D[(col_C == col_A) & (col_B == 'no')].iloc[0]
        mirror = col_D[(col_C == col_A) & (col_B == 'yes')].iloc[0]
        alea = col_D[(col_C != col_A) & (col_B == 'no')].iloc[0]

        results.append({
            "file": file_name,
            "sheet": sheet,
            "Rsequence": sequence,
            "Rmirror": mirror,
            "Ralea": alea
        })

# Final dataframe
result_df = pd.DataFrame(results)

# Save to Excel
result_df.to_excel("Recognition_data.xlsx", index=False)