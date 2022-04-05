from xfoil import XFoil
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


# TODO: Compare the new airfoils (100 points up and low) to the original
# TODO: Use some kind of MSE
# TODO: Check Cl, Cd, Cm for a reasonable amount of AoA (3?)


class AirfoilConstructor:

    def __init__(self, airfoil_):
        self.airfoil = airfoil_
        self.x = airfoil_[:, 0]
        self.y = airfoil_[:, 1]
        self.n_coords = len(self.y)


def Xfoil_Runner(path, name, Re, n_crit, aoa=None, cl=None):

    if '.dat' in name:
        pass
    else:
        name += '.dat'
    airfoil = np.loadtxt(f"{path}/{name}", usecols=[0, 1], skiprows=1)
    airfoil_obj = AirfoilConstructor(airfoil)

    xf = XFoil()
    xf.print = False
    xf.airfoil = airfoil_obj
    xf.filter(factor=0.2)
    xf.repanel()

    xf.Re = Re
    xf.max_iter = 100
    xf.n_crit = n_crit

    if isinstance(aoa, (np.ndarray, list)):
        a_start = aoa[0]
        a_end = aoa[-1]
        step = (aoa[1] - aoa[0]) / 2
        a, cl, cd, cm, cp = xf.aseq(a_start, a_end, step)
        return a, cl, cd, cm, cp
    else:
        cl, cd, cm, cp = xf.a(aoa)
        return cl, cd, cm, cp


if __name__ == "__main__":
    # df = pd.read_excel("Airfoil_Database_Errors.xlsx", index_col=0)
    #
    # airfoils = os.listdir("newAirfoils")
    #
    # print(df.Names.values)
    # print(airfoils)
    # print("test")
    # for airfoil in airfoils:
    #     name = airfoil.replace(".dat", "")
    #     if name not in df.Names.values:
    #         os.remove(f"newAirfoils/{airfoil}")

    # df = pd.read_excel("Airfoil_Database_original.xlsx", index_col=0)
    # df2 = pd.read_excel("Airfoil_Database_new.xlsx", index_col=0)
    # df2 = df2[df2.names.isin(df.names)]
    #
    # print("Original dataframe shape with original airfoils: ", df.shape)
    # print("Original dataframe shape with new airfoils: ", df2.shape)
    #
    # df2_nafree = df2.dropna(axis=0, how='any')
    # only_na = df2[~df2.index.isin(df2_nafree.index)]
    #
    # print("Dropped airfoils: ", only_na.shape)
    # print("New dataframe shape with new airfoils: ", df2_nafree.shape)
    #
    # df = df[~df.names.isin(only_na.names)]
    #
    # df_nafree = df.dropna(axis=0, how='any')
    # only_na_original = df[~df.index.isin(df_nafree.index)]
    #
    # df2_nafree = df2_nafree[~df2_nafree.names.isin(only_na_original.names)]
    #
    # print("Original dataframe shape with dropped airfoils and dropped nas: ", df_nafree.shape)
    # print("New dataframe shape with dropped airfoils and dropped nas: ", df2_nafree.shape)

    #
    # df_nafree.to_excel("Airfoil_Database_original_cleared.xlsx")
    # df2_nafree.to_excel("Airfoil_Database_newx_cleared.xlsx")

    # error_cl = 100 * (df_nafree.Cl - df2_nafree.Cl) / df_nafree.Cl
    # error_cd = 100 * (df_nafree.Cd - df2_nafree.Cd) / df_nafree.Cd
    # error_cm = 100 * (df_nafree.Cm - df2_nafree.Cm) / df_nafree.Cm
    # frame = {'Cl Error': error_cl, "Cd Error": error_cd, "Cm Error": error_cm, "Names": df_nafree.names}
    # errors_df = pd.DataFrame(frame)
    # print("Errors dataframe shape: ", errors_df.shape)
    #
    # errors_df = errors_df[errors_df['Cl Error'].abs() <= 100]
    # errors_df = errors_df[errors_df['Cd Error'].abs() <= 100]
    # errors_df = errors_df[errors_df['Cm Error'].abs() <= 100]
    #
    # print("Errors (edited) dataframe shape: ", errors_df.shape)

    # errors_df['Cl Error'].plot.scatter(x='names', y='Cl Error')
    # plt.show()
    # errors_df.to_excel("Airfoil_Database_Errors.xlsx")

    # print(only_na.head(50))
    ############################################################################################

    this_path = os.getcwd()
    airfoils_dir = os.path.join(this_path, "newAirfoils_cosine2")
    number_of_airfoils = len(os.listdir(airfoils_dir))
    names = [foil.replace('.dat', '') for foil in os.listdir(airfoils_dir)]

    cl_all = []
    cd_all = []
    cm_all = []
    cp_all = []

    for i, airfoil in enumerate(os.listdir(airfoils_dir)):
        print(airfoil)
        cl, cd, cm, cp = Xfoil_Runner(airfoils_dir, airfoil, Re=500000, n_crit=9, aoa=0)

        cl_all.append(cl)
        cd_all.append(cd)
        cm_all.append(cm)
        cp_all.append(cp)

    print("\n###########################")
    df = pd.DataFrame(np.array([cl_all, cd_all, cm_all, cp_all]).T, columns=['Cl', 'Cd', 'Cm', 'Cp'])
    df['Cl_Cd'] = df['Cl'] / df['Cd']
    print("Not converged rows:",  df.isnull().any(axis=1).sum())

    df['names'] = names
    print(df.head(50))

    df.to_excel("Airfoil_Database_new.xlsx")