import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import interpolate
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline


def split_upper_lower_curves(x_Points, y_Points):
    # print(min(x_Points))
    # # print(np.where(x_Points <= 5e-05))
    zero_index = np.where(x_Points == min(x_Points))[0][0]
    # print(zero_index)
    # half_index = len(x_Points) // 2
    upper_indexes = np.arange(0, zero_index)
    lower_indexes = np.arange(zero_index, len(x_Points))

    upper_curve = (x_Points[upper_indexes], y_Points[upper_indexes])
    lower_curve = (x_Points[lower_indexes], y_Points[lower_indexes])

    return upper_curve, lower_curve


def split_te_le_body(x_Points, y_Points):
    pass


def check_for_duplicates(x_Points):
    u, c = np.unique(x_Points, return_counts=True)
    dup = u[c > 1]
    if len(dup) > 0:
        index_dup = np.where(x_Points == dup[0])[0][0]
        x_Points[index_dup] += 0.000001


def create_new_xpoints(no_Points):
    finest_edges = no_Points // 20
    coarser_edges = no_Points // 3
    generic = no_Points - (2 * finest_edges) - (2 * coarser_edges)
    xnew_le_e = np.linspace(0, 0.02, finest_edges)
    xnew_le = np.linspace(0.02, 0.2, coarser_edges)
    xnew_generic = np.linspace(0.2, 0.8, generic)
    xnew_te = np.linspace(0.8, 0.98, coarser_edges)
    xnew_te_e = np.linspace(0.98, 1.00, finest_edges)
    xnew = np.concatenate([xnew_le_e, xnew_le, xnew_generic, xnew_te, xnew_te_e])

    return xnew


def create_new_xpoints_cosine(x, y, N=100):
    R = (x.max() - x.min()) / 2  # radius of the circle
    x_center = (x.max() + x.min()) / 2  # x-coord of the center
    # define x-coord of the circle points
    x_circle = x_center + R * np.cos(np.linspace(0.0, 2 * np.pi, N + 1))

    x_ends = np.copy(x_circle)  # projection of the x-coord on the surface
    y_ends = np.empty_like(x_ends)  # initialization of the y-coord Numpy array

    x, y = np.append(x, x[0]), np.append(y, y[0])  # extend arrays using numpy.append

    # computes the y-coordinate of end-points
    I = 0
    for i in range(N):
        while I < len(x) - 1:
            if (x[I] <= x_ends[i] <= x[I + 1]) or (x[I + 1] <= x_ends[i] <= x[I]):
                break
            else:
                I += 1
        a = (y[I + 1] - y[I]) / (x[I + 1] - x[I])
        b = y[I + 1] - a * x[I + 1]
        y_ends[i] = a * x_ends[i] + b
    y_ends[N] = y_ends[0]

    return x_ends, y_ends


def testing_cosine(x, N=100):

    Neff = N // 2
    startPoint = min(x)
    endPoint = max(x)
    midPoint = (endPoint - startPoint) / 2
    angleInc = np.pi / (Neff - 1)

    curAngle = angleInc
    xNew_temp = np.empty(Neff)
    xNew_temp[0] = startPoint
    xNew_temp[-1] = endPoint

    for i in range(1, Neff - 1):
        xNew_temp[i] = startPoint + midPoint * (1 - np.cos(curAngle))
        curAngle = curAngle + angleInc
    #

    xNew = np.empty(N)
    xNew[:Neff] = xNew_temp[::-1]
    xNew[Neff:] = xNew_temp

    return xNew


def create_new_files(method):
    airfoils = os.listdir("coord_seligFmt")

    for airfoil in airfoils:
        x, y = np.loadtxt(f"coord_seligFmt/{airfoil}", usecols=[0, 1], unpack=True, skiprows=1)
        up, low = split_upper_lower_curves(x, y)

        check_for_duplicates(up[0])
        check_for_duplicates(low[0])

        f_up = interpolate.interp1d(up[0], up[1], kind="cubic", fill_value="extrapolate")
        f_low = interpolate.interp1d(low[0], low[1], kind="cubic", fill_value="extrapolate")

        path = None
        xnew = None
        if method == "linear":
            xnew = create_new_xpoints(100)
            path = "newAirfoils"
            y_new_up = f_up(xnew)
            y_new_low = f_low(xnew)
        elif method == "cosine":
            xnew, ynew = create_new_xpoints_cosine(x, y, N=2 * 100)
            path = "newAirfoils_cosine"
        elif method == "cosine2":
            xnew = testing_cosine(x, N=2 * 100)
            y_new_up = f_up(xnew)
            y_new_low = f_low(xnew)

            path = "newAirfoils_cosine2"

        with open(f"{path}/{airfoil}", 'w+') as f:
            if method == "linear":
                for i in range(xnew.shape[0] - 1, -1, -1):
                    row_str = f"{xnew[i]} {y_new_up[i]}\n"
                    f.write(row_str)
                for i in range(xnew.shape[0]):
                    row_str = f"{xnew[i]} {y_new_low[i]}\n"
                    f.write(row_str)
            elif method == "cosine":
                for i in range(xnew.shape[0] // 2 + 1):
                    row_str = f"{xnew[i]} {ynew[i]}\n"
                    f.write(row_str)
                for i in range(xnew.shape[0] // 2 - 1, xnew.shape[0]):
                    row_str = f"{xnew[i]} {ynew[i]}\n"
                    f.write(row_str)
            elif method == "cosine2":
                for i in range(xnew.shape[0] // 2):
                    row_str = f"{xnew[i]} {y_new_up[i]}\n"
                    f.write(row_str)
                for i in range(xnew.shape[0] // 2, xnew.shape[0]):
                    row_str = f"{xnew[i]} {y_new_low[i]}\n"
                    f.write(row_str)

        print(airfoil)
        # except Exception:
        #     pass

        # x1, y1 = np.loadtxt(f"coord_seligFmt/{airfoil}", usecols=[0, 1], unpack=True, skiprows=1)
        # x, y = np.loadtxt(f"newAirfoils_cosine2/{airfoil}", usecols=[0, 1], unpack=True)
        #
        # plt.scatter(x1, y1, color='r', s= 5)
        # plt.scatter(x, y, color='b', s=5)
        # plt.axis('equal')
        # plt.show()


if __name__ == "__main__":
    create_new_files(method="cosine2")
    # x_, y_ = np.loadtxt(f"newAirfoils_cosine/a18.dat", dtype=float, usecols=[0, 1], unpack=True, skiprows=1)
    # x, y = np.loadtxt(f"coord_seligFmt/a18.dat", dtype=float, usecols=[0, 1], unpack=True, skiprows=1)
    # xNew = testing_cosine(x, N=100)
    # xNew2, _ = create_new_xpoints_cosine(x, y, N=100)

    # print(x)
    # print(xNew)
    # print(xNew2)

    #
    # plt.plot(x_, y_, color='b', alpha=0.4)
    # plt.plot(x, y, color='r')
    # plt.axis("equal")
    # plt.show()


# # x1, y1 = np.loadtxt(f"newAirfoils/mh62.dat", usecols=[0, 1], unpack=True, skiprows=1)
# # plt.plot(x1, y1, 'r')
# plt.plot(x, y, 'b')
# plt.axis('equal')
# plt.show()

# plt.scatter(xnew, y_new_up, s=10, label='fine')
# plt.scatter(xnew2, f_up(xnew2), s=10, label='coarse')
# plt.legend()

# # print(xnew)
# plt.scatter(xnew)
# plt.plot(x, y, '--')
# plt.plot(xnew, y_new_up, '-', xnew, y_new_low, '-')
# plt.axis('equal')
# plt.show()

# # f2 = interpolate.interp1d(x, y, kind='cubic')
# xnew = np.linspace(0, 1, 64)
#
# plt.plot(x, y, 'o', x, f(x), '-')
# plt.legend(['data', 'linear'], loc='best')

