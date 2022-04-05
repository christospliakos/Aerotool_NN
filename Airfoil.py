import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import os
from scipy import interpolate
from xfoil import XFoil
import scipy.special
from scipy.optimize import minimize


# TODO: Find leading edge radius
# TODO: Find if trailing edge is closed
# TODO: Close trailing edge if wanted
# TODO: Reformat with new x (cosine/linear)
# TODO: Return bezier control points that match the airfoil as well as possible
# TODO: Get Polars from XFOIL?

class AirfoilBezier:

    def __init__(self, noPoints, originalLE, originalTE):
        self.noPoints = noPoints
        self.LeadingEdge = originalLE
        self.TrailingEdge = originalTE

    def __get_order(self, points):
        if isinstance(points, list):
            self.order = len(points)
        elif isinstance(points, np.ndarray):
            self.order = points.shape[0]

    def __bernstein_poly(self, t, i):
        return scipy.special.comb(self.order, i) * (t ** i) * ((1 - t) ** (self.order - i))

    def __bezier_curve(self, points, curveFlag):
        self.t = np.linspace(0, 1, self.noPoints)
        curve = []
        for t in self.t:
            B_x = 0
            if curveFlag == "Upper":
                B_y = self.LeadingEdge[1]
            elif curveFlag == "Lower":
                B_y = self.LeadingEdge[3]
            # TODO: make this with numpy (dot etc)
            for i in range(self.order):
                B_x += self.__bernstein_poly(t, i) * points[i][0]
                B_y += self.__bernstein_poly(t, i) * points[i][1]
            curve.append((B_x, B_y))

        return curve

    def __construct_airfoil(self):
        self.airfoil_x = np.array([])
        self.airfoil_y = np.array([])

        self.airfoil_x = np.concatenate((self.airfoil_x, self.uppercurve[0]), axis=None)
        self.airfoil_x = np.concatenate((self.airfoil_x, self.lowercurve[0][::-1]), axis=None)

        self.airfoil_y = np.concatenate((self.airfoil_y, self.uppercurve[1]), axis=None)
        self.airfoil_y = np.concatenate((self.airfoil_y, self.lowercurve[1][::-1]), axis=None)

    def get_bezier_airfoil(self, xPositions, upperfoil, lowerfoil):

        customUpperFoil = list(map(list, zip(xPositions, upperfoil)))
        customLowerFoil = list(map(list, zip(xPositions, lowerfoil)))

        upperfoil = [[0, self.LeadingEdge[1]],
                     *customUpperFoil,
                     [self.TrailingEdge[0], self.TrailingEdge[1]]]

        lowerfoil = [[0, self.LeadingEdge[3]],
                     *customLowerFoil,
                     [self.TrailingEdge[2], self.TrailingEdge[3]]]

        self.__get_order(upperfoil)
        upperCtrlPoints = upperfoil[::-1]
        lowerCtrlPoints = lowerfoil[::-1]

        self.uppercurve = list(zip(*(self.__bezier_curve(upperCtrlPoints, curveFlag="Upper"))))
        self.lowercurve = list(zip(*(self.__bezier_curve(lowerCtrlPoints, curveFlag="Lower"))))

        self.__construct_airfoil()

        return self.airfoil_x, self.airfoil_y


class Airfoil:

    def __init__(self, x: Union[list, np.ndarray] = None, y: Union[list, np.ndarray] = None, airfoilName: str = None):
        if (x is None or y is None) and airfoilName:
            self.airfoilName = airfoilName
            self.__get_coors_from_name()
        elif not airfoilName and (x is not None and y is not None):
            self.x = np.array(x) if isinstance(x, list) else x
            self.y = np.array(y) if isinstance(y, list) else y
            # We need to normalize in order to compare the same things.
            self.__normalize_coors_based_on_chord()
            self.__check_validity_of_preloaded_coors()

        self.__set_main_characteristics()

        # self.leadingEdgeRadius = self.__get_leading_edge_radius()

    def __get_coors_from_name(self):
        """
        Searches inside the available database for airfoils (UUIC) to find the one that the user wants.
        If the airfoil is available then it stores the coordinates, else it raises an error.
        :return:
        """
        availableFoils = os.listdir("coord_seligFmt")

        if ".dat" not in self.airfoilName:
            self.airfoilName += '.dat'

        if self.airfoilName not in availableFoils:
            raise NameError(f"The specific airfoil: {self.airfoilName} is not available into the database, "
                            f"try inserting it with coordinates.")

        self.x, self.y = np.loadtxt(f"coord_seligFmt/{self.airfoilName}", unpack=True, usecols=[0, 1], skiprows=1)

    def __check_validity_of_preloaded_coors(self):
        # If sorted then it is in the wrong order
        if np.all(self.x[:-1] <= self.x[1:]) or np.all(self.y[:-1] <= self.y[1:]):
            raise Exception("The airfoil coordinates are not in the correct order/form.")

        # This means that the airfoil coords start/end from/at the trailing edge
        if (abs(self.x[0] - 1.00) <= 1e-6) and (abs(self.y[0] - 1.00) <= 1e-6):
            pass
        else:
            raise Exception("The airfoil coordinates are not in the correct order/form. They should start and finish"
                            "at the trailing edge.")

    def __normalize_coors_based_on_chord(self):
        """ Normalizes the airfoil dimension by the chord length.
            It normalizes both X and Y with the same factor.

            The change happens IN PLACE.
        """
        chordLine = self.x.max() - self.x.min()
        self.x / chordLine
        self.y / chordLine

    def __get_upper_lowers_coors(self):
        """
        Finds the index where the coordinates change from suction side to pressure side.
        Firstly it assumes that the index where the change of sides happen is right at the middle, but if this fails
        it uses numpy to find the correct index.
        Then it returns the X and Y coordinates of both curves.
        :return:
        """
        # print(self.x.shape)
        # print(self.x.shape[0])
        midIndex = self.x.shape[0] // 2

        if self.x[midIndex] <= 1e-5:
            pass
        else:
            midIndex = np.argmin(self.x)
            print("test")

        # print(self.x[midIndex:])
        # print(self.x[:midIndex])

        upper = [self.x[:midIndex + 1][::-1], self.y[:midIndex + 1][::-1]]
        lower = [self.x[midIndex:], self.y[midIndex:]]

        upper[0], removeIndexUpper = self.__check_duplicates_in_array(upper[0], treatment="remove")
        upper[1], _ = self.__check_duplicates_in_array(upper[1], treatment="remove", indexToRemove=removeIndexUpper)

        lower[0], removeIndexLower = self.__check_duplicates_in_array(lower[0], treatment="remove")
        lower[1], _ = self.__check_duplicates_in_array(lower[1], treatment="remove", indexToRemove=removeIndexLower)

        self.upper_coors, self.lower_coors = upper, lower

        # if self.upper_coors[0].shape != self.lower_coors[0].shape:
        #     # print(self.upper_coors[0].shape)
        #     # print(self.lower_coors[0].shape)
        #     # print("test")
        #     self.create_new_spacing(N=min([self.upper_coors[0].shape, self.lower_coors[0].shape])[0],
        #                             minimumX=min(self.upper_coors[0]),
        #                             inplace=True)

    def __get_thickness_line(self):
        """
        It calculates the thickness distribution along the chord
        :return:
        """
        xSpacing = np.linspace(0, 1, 101)
        upper = np.interp(xSpacing,
                          self.upper_coors[0],
                          self.upper_coors[1]).T
        lower = np.interp(xSpacing,
                          self.lower_coors[0],
                          self.lower_coors[1]).T

        return self.upper_coors[0], self.upper_coors[1] - self.lower_coors[1]

    def __get_maximum_thickness_and_position(self):
        """
        Calculates the maximum thickness (dimensionless) of the airfoil as well as the position of this in terms of
        percentage.
        :return:
        """
        thicknessDistribution = self.thicknessLine[1]
        maxThicknessIndex = np.argmax(thicknessDistribution)
        return np.max(thicknessDistribution), 100 * self.upper_coors[0][maxThicknessIndex]

    def __get_camber_line(self):
        """
        It calculates the camber line of the airfoil by finding the middle point for each section.
        :return: Returns a tuple of two np.ndarray for the X coordinate and the camber line.
        """
        # xSpacing = np.linspace(0, 1, 101)
        # upper = np.interp(xSpacing,
        #                   self.upper_coors[0],
        #                   self.upper_coors[1]).T
        # lower = np.interp(xSpacing,
        #                   self.lower_coors[0],
        #                   self.lower_coors[1]).T

        return self.upper_coors[0], (self.upper_coors[1] + self.lower_coors[1]) / 2

    def __get_maximum_camber_and_position(self):
        """
        It calculates and returns the maximum camber as well as the position of this.
        :return:
        """
        maxCamberIndex = np.argmax(self.camberLine[1])
        return np.max(self.camberLine[1]), 100 * self.upper_coors[0][maxCamberIndex]

    def __get_leading_edge_radius(self):
        # TODO: FINISH THIS SHIT
        """
        Calculates the approximate leading edge radius
        :return: Leading edge radius relatively with the chord length
        """
        if np.where(self.camberLine[0] <= 0.005 * self.chord)[0].shape[0] <= 1:
            x_interest_index = 1
            slope = (self.camberLine[1][1] - self.camberLine[1][0]) / (
                    self.camberLine[0][1] - self.camberLine[0][0])
        else:
            x_interest_index = np.where(self.camberLine[0] <= 0.005 * self.chord)[0][-1]
            slope = (self.camberLine[1][x_interest_index + 1] - self.camberLine[1][x_interest_index - 1]) / (
                    self.camberLine[0][x_interest_index + 1] - self.camberLine[0][x_interest_index - 1])

        xCoordRange = np.linspace(0, self.upper_coors[0][x_interest_index + 5], 100)
        tangentLine = xCoordRange * slope + self.upper_coors[1][0]

        crossPoint = np.subtract(self.camberLine[1][:x_interest_index + 5], tangentLine)
        print(crossPoint)

        return None

    def __set_main_characteristics(self):
        self.chord = round(self.x.max() - self.x.min(), 2)

        # plt.plot(self.x, self.y, '-*', label="Original")

        self.__get_upper_lowers_coors()
        self.camberLine = self.__get_camber_line()
        self.thicknessLine = self.__get_thickness_line()
        self.maximumThickness, self.maximumThicknessLocation = self.__get_maximum_thickness_and_position()
        self.maximumCamber, self.maximumCamberLocation = self.__get_maximum_camber_and_position()

    @staticmethod
    def __check_duplicates_in_array(pointArray: np.ndarray, treatment: str = "keep", indexToRemove: int = None):
        """
        Checking if an array has duplicate points in it. By having duplicates into an array we cannot perform
        the interpolation to find the upper/lower curves.
        :param pointArray: The array that needs treatment
        :param treatment: There are two options, either keep or remove. By keeping the point we are adding to one of
                          the duplicates an infinitesimal amount. By removing we are popping the element. This happens
                          ONLY if the duplicate points are next to each other.
        :param indexToRemove: If given then this index will be specifically removed and the whole procedure will be
                              skipped
        :return: Returns the treated array.
        """
        u, c = np.unique(pointArray, return_counts=True)
        dup = u[c > 1]
        if indexToRemove is not None:
            pointArray = np.delete(pointArray, indexToRemove)
            return pointArray, None

        index_dup = None
        if len(dup) > 0:
            index_dups = np.where(pointArray == dup[0])[0]
            if (abs(index_dups[0] - index_dups[1]) == 1) and index_dups[0] in [0, 1, 2, 97, 98, 99, 100]:
                index_dup = np.where(pointArray == dup[0])[0][0]
                if treatment == "keep":
                    if pointArray[index_dup] >= 1.00:
                        pointArray[index_dup] -= 0.000001
                    else:
                        pointArray[index_dup] += 0.000001
                elif treatment == "remove":
                    pointArray = np.delete(pointArray, index_dup)
        return pointArray, index_dup

    def create_new_spacing(self, N: Union[int, float] = 100,
                           minimumX: float = 0.00,
                           desiredSpacing: Union[list, np.ndarray] = None,
                           inplace: bool = True):
        """
        Creates a new spacing for x points. That means that the upper and lower curves are being generated again.
        By default it uses cosine scheme, unless the user provides a specific X spacing.
        :param minimumX: Pass the minimum X that the spacing will start from.
        :param N: The number of points wanted for each curve (the total X/Y coordinate will have 2 * N points)
        :param desiredSpacing: Expects a list/nparray with the desired X spacing ex: [25, 50, 25]. The locations where
                               this happens is at 0 -> 0.2, 0.2 -> 0.8, 0.8 -> 1.0. The total amount of points must
                               match N
        :param inplace: If True the current airfoil object is getting affected, if False a tuple of X and Y is being
                        returned.
        :return: Returns new X/Y np.arrays if inplace is True
        """
        if desiredSpacing is None:
            deltaPhi = np.pi / (2 * (N - 1))
            steps = np.cos(np.arange(0, N) * deltaPhi)
            xSpacing = 1 - steps
        else:
            desiredSpacing = np.array(desiredSpacing) if isinstance(desiredSpacing, list) else desiredSpacing
            if desiredSpacing.sum() != N:
                raise Warning("The number of points in desiredSpacing doesn't sum up to N")

            xSpacing = np.hstack((np.linspace(0, 0.2, desiredSpacing[0]),
                                  np.linspace(0.2, 0.8, desiredSpacing[1]),
                                  np.linspace(0.8, 1.0, desiredSpacing[2])))

        newY = self.create_new_y_curves(xPoints=xSpacing)
        newX = np.concatenate((xSpacing[::-1], xSpacing), axis=0)
        if not inplace:
            return newX, newY
        else:
            self.x = newX
            self.y = newY
            self.__set_main_characteristics()

    def create_new_y_curves(self, xPoints: np.ndarray):
        """
        It uses an 1d interpolation and more specifically the cubic interpolation to create a model/approximation
        for the upper and lower curves. After that it uses the xPoints to re-create the curves
        :param xPoints: The desired X spacing
        :return: Concatenated the y curves as one.
        """

        upperCoordsCleanX, _ = self.__check_duplicates_in_array(np.copy(self.upper_coors[0]), treatment="keep")
        lowerCoordsCleanX, _ = self.__check_duplicates_in_array(np.copy(self.lower_coors[0]), treatment="keep")

        f_up = interpolate.interp1d(upperCoordsCleanX, self.upper_coors[1], kind="cubic", fill_value="extrapolate")
        f_low = interpolate.interp1d(lowerCoordsCleanX, self.lower_coors[1], kind="cubic", fill_value="extrapolate")

        newYUpper = f_up(xPoints)[::-1]
        newYLower = f_low(xPoints)

        return np.concatenate((newYUpper, newYLower), axis=0)

    def get_polars(self,
                   Reynolds: float = 1e6,
                   Mach: float = 0,
                   nCrit: int = 9,
                   angleOfAttack: Union[int, list, np.ndarray] = 0,
                   ):
        # TODO: Finish this
        pass

    def get_bezier_control_points(self,
                                  noOfPoints: int = 100,
                                  noOfControlPoints: int = 6):

        def get_loss(controlPoints, returnAirfoil=False):
            upperCtrlPoints = controlPoints[:noOfControlPoints - 2]
            lowerCtrlPoints = controlPoints[noOfControlPoints - 2:]

            airfoil_x, airfoil_y = Bezier.get_bezier_airfoil(xPositions, upperCtrlPoints, lowerCtrlPoints)

            loss = np.sum((np.subtract(self.y, airfoil_y) ** 2))

            if returnAirfoil:
                return airfoil_x, airfoil_y

            # upperLoss = np.sum(np.square(np.subtract(originalAirfoil[0], newAirfoil[0])))
            return loss

        ################################################################
        # TODO: Create initial guess with the number of control points given
        xPositions = [0.1, 0.25, 0.50, 0.75]
        yUpperCurve = [0.12785708, 0.14146167, 0.12797651, 0.15479387]
        yLowerCurve = [-0.01096851, -0.05085217, 0.14821516, 0.14453813]
        ctrlPoints = np.array([*yUpperCurve, *yLowerCurve])
        ################################################################

        Bezier = AirfoilBezier(noPoints=noOfPoints,
                               originalLE=[self.upper_coors[0][0], self.upper_coors[1][0],
                                           self.lower_coors[0][0], self.lower_coors[1][0]],
                               originalTE=[self.upper_coors[0][-1], self.upper_coors[1][-1],
                                           self.lower_coors[0][-1], self.lower_coors[1][-1]])

        res = minimize(get_loss, ctrlPoints, method="BFGS", tol=1e-6, options={"disp": True})
        # res = scipy.optimize.shgo(get_loss, bounds=[(-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1)])

        airfoilX, airfoilY = get_loss(res.x, returnAirfoil=True)
        print(self.upper_coors[1][0], *res.x[:4], self.upper_coors[1][-1])
        plt.plot([0, 0.1, 0.25, 0.50, 0.75, 1.0], [self.upper_coors[1][0], *res.x[:4], self.upper_coors[1][-1]], '-o')
        plt.plot([0, 0.1, 0.25, 0.50, 0.75, 1.0], [self.lower_coors[1][0], *res.x[4:], self.lower_coors[1][-1]], '-o')
        # plt.plot((0, 1), (self.upper_coors[1][0], self.upper_coors[1][-1]), '-o')
        # plt.plot((0, 1), (self.lower_coors[1][0], self.lower_coors[1][-1]), '-o')
        self.plot_airfoil([airfoilX, airfoilY])


        print("test")

    def save_airfoil(self, path: str, name: str):
        """

        :param path:
        :param name:
        :return:
        """
        newArray = np.vstack((self.x, self.y)).T
        np.savetxt(f"{path}/{name}.dat", newArray)

    def plot_airfoil(self, otherAirfoil=None):
        """

        :param otherAirfoil:
        :return:
        """
        # fig, ax = plt.subplots(figsize=(10, 4))
        if otherAirfoil is None:
            plt.plot(self.x, self.y, 'o-', ms=4, color='green', alpha=0.2)
            plt.plot(self.upper_coors[0], self.upper_coors[1], 'r--', label='Upper Surface')
            plt.plot(self.lower_coors[0], self.lower_coors[1], 'k--', label='Lower Surface')
            plt.plot(self.camberLine[0], self.camberLine[1], 'y--', label='Camber Line')
            plt.vlines(x=self.maximumThicknessLocation / 100,
                       ymin=self.lower_coors[1][np.argmax(self.upper_coors[1])],
                       ymax=self.upper_coors[1].max(), color='r',
                       label=f'Maximum Thickness {round(100 * self.maximumThickness, 2)}%'
                             f' at {round(self.maximumThicknessLocation, 2)}%')
        else:
            plt.plot(self.x, self.y, 'o-', ms=4, label='Original')
            plt.plot(otherAirfoil[0], otherAirfoil[1], color='r', label='New Airfoil')

        plt.axis("equal")
        plt.grid()
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # 1st case with airfoil name
    airfoil_name = "naca0015"
    airfoil = Airfoil(airfoilName=airfoil_name)

    # 2nd case with airfoil coords

    # x_, y_ = np.loadtxt("coord_seligFmt/a18.dat", unpack=True, skiprows=1, usecols=[0, 1])
    # x_ = [0.0, 0.0125, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9,
    #       0.95, 1.0, 1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.075,
    #       0.05, 0.025, 0.0125, ]
    # y_ = [y_[i] for i in range(y_.shape[0])]
    #
    # print(x_, y_)
    # airfoil = Airfoil(x=x_, y=y_)

    # Visualizing
    # airfoil.plot_airfoil()
#
    airfoil.create_new_spacing(N=100, inplace=True)

    # airfoil.get_bezier_control_points(noOfPoints=100,
    #                                   noOfControlPoints=6)

    # spacing = np.array([80, 40, 80])
    # airfoil.create_new_spacing(N=200, desiredSpacing=spacing, inplace=True)

    # airfoil.save_airfoil("TestingAirfoils", "clarkyNew")
    #
    airfoil.plot_airfoil()

    ###############

