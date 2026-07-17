import numpy as np
import pandas as pd
import pytest

from warfarin_dose.data import REQUIRED_COLUMNS


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    rows = []
    for site in range(6):
        for sample in range(8):
            weight = 50.0 + sample
            cyp2c9 = ("*1/*1", "*1/*2", "*1/*3")[sample % 3]
            vkorc1 = ("G/G", "A/G", "A/A")[sample % 3]
            row = {column: np.nan for column in REQUIRED_COLUMNS}
            row.update(
                {
                    "PharmGKB Subject ID": f"subject-{site}-{sample}",
                    "PharmGKB Sample ID": f"sample-{site}-{sample}",
                    "Project Site": f"site-{site}",
                    "Gender": "male" if sample % 2 else "female",
                    "Race (Reported)": "source-shaped race",
                    "Race (OMB)": "source-shaped race",
                    "Age": "50 - 59",
                    "Height (cm)": 160.0 + sample,
                    "Weight (kg)": weight,
                    "Indication for Warfarin Treatment": "atrial fibrillation",
                    "Diabetes": 0,
                    "Congestive Heart Failure and/or Cardiomyopathy": 0,
                    "Valve Replacement": 0,
                    "Simvastatin (Zocor)": 0,
                    "Atorvastatin (Lipitor)": 0,
                    "Fluvastatin (Lescol)": 0,
                    "Lovastatin (Mevacor)": 0,
                    "Pravastatin (Pravachol)": 0,
                    "Rosuvastatin (Crestor)": 0,
                    "Cerivastatin (Baycol)": 0,
                    "Amiodarone (Cordarone)": 0,
                    "Carbamazepine (Tegretol)": 0,
                    "Phenytoin (Dilantin)": 0,
                    "Rifampin or Rifampicin": 0,
                    "Target INR": 2.5,
                    "Estimated Target INR Range Based on Indication": "2.0 - 3.0",
                    "Subject Reached Stable Dose of Warfarin": 1,
                    "Therapeutic Dose of Warfarin": 10.0
                    + 2.0 * site
                    + 0.1 * weight
                    - 3.0 * (vkorc1 == "A/A"),
                    "INR on Reported Therapeutic Dose of Warfarin": 2.5,
                    "Current Smoker": 0,
                    "CYP2C9 consensus": cyp2c9,
                    "VKORC1 -1639 consensus": vkorc1,
                    "Comments regarding Project Site Dataset": "synthetic fixture",
                }
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
