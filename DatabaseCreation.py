import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from XFOIL_Compare import Xfoil_Runner


def clean_data(angleOfAttack, liftCoefficient, dragCoefficient, momentCoefficient):

    """
    Checks if any of the coefficient lists is totally empty/full of nans and skips it as it did not converge.
    If the coefficients pass the first test then pandas try to interpolate through the available values while
    simultaneously it drops all the NaN rows.

    :param angleOfAttack: Expects an array/list of angles
    :param liftCoefficient: Expects an array of Cl
    :param dragCoefficient: Expects an array of Cd
    :param momentCoefficient: Expects an array of Cm
    :return: Cleaned dataframe from NaNs and interpolated.
    """

    dataArray = np.array([angleOfAttack, liftCoefficient, dragCoefficient, momentCoefficient]).T

    df = pd.DataFrame(dataArray, columns=['AoA', 'Cl', 'Cd', 'Cm'])

    if df.Cl.isnull().all() or df.Cd.isnull().all() or df.Cm.isnull().all():
        return False

    try:
        df_new = df.interpolate(method='quadratic').dropna(axis=0, how='any')
    except ValueError:
        return False
    return df_new


def create_xlsx(dataFrame, path):
    path = path + '.xlsx'
    dataFrame.to_excel(path, index=False)


this_path = os.getcwd()
airfoils_dir = os.path.join(this_path, "coord_seligFmt")
number_of_airfoils = len(os.listdir(airfoils_dir))
names = [foil.replace('.dat', '') for foil in os.listdir(airfoils_dir)]

aoa = np.arange(-10, 15, 0.5)
reynolds = 500000
for i, airfoil in enumerate(os.listdir(airfoils_dir)):
    print(airfoil)

    a, cl, cd, cm, cp = Xfoil_Runner(airfoils_dir, airfoil, Re=reynolds, n_crit=9, aoa=aoa)

    cleaned_df = clean_data(a, cl, cd, cm)
    if not isinstance(cleaned_df, bool):
        create_xlsx(cleaned_df, path=f"Database/Re_{reynolds}/{airfoil}")

