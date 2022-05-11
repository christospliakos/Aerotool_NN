import numpy as np
import matplotlib.pyplot as plt
from math import *

"""Global Variables"""
W = 1410958.48  # lb
Wf = 749571.691
S = 9741.34  # ft^2
b = 27.5591  # ft , in to wingspan
AR = (b ** 2) / S
e_span = 0.85  # oswald factor empeirika
Cd0 = 0.029
K = 0.04
k3 = 1 / (3.14 * e_span * AR)
k1 = K - k3
Lamda = 22  # moires
Tav = 6 * 51593.65169  # lbf
Tclimb = 6 * 39341
ct = 0.546  # lb/lbf/h
Vrange = np.arange(150, 2500, 50)

#######
Cl_Cd_max = (1 / (4 * Cd0 * K)) ** (1 / 2)
print("Το μέγιστο Cl/Cd είναι: " + str(Cl_Cd_max))

Cl_1_2_Cd_max = (3 / 4) * ((1 / (3 * K * (Cd0 ** 3))) ** (1 / 4))
print("Το μέγιστο Cl(1/2) / Cd είναι: " + str(Cl_1_2_Cd_max))


def Trequired(altitude, mission='cruise'):
    if mission == "climb":
        Tav_ = Tclimb
    else:
        if altitude == 0:
            rho = 2.3769 * (10 ** (-3))  # slug/ft^3
            Tav_ = Tav
        elif altitude == 30000:
            rho = 8.9068 * (10 ** (-4))  # slug/ft^3
            Tav_ = Tav * ((rho / 0.002377) ** (0.9))
        elif altitude == 10000:
            rho = 1.7556 * (10 ** (-3))  # slug/ft^3
            Tav_ = Tav * ((rho / 0.002377) ** (0.9))
    Tr_list = np.array([])

    for V in Vrange:
        Cl = (2 * W) / (rho * (V ** 2) * S)
        Cd = Cd0 + (K * (Cl ** 2))
        Tr = (0.5 * rho * (V ** 2) * S) * Cd
        Tr_list = np.append(Tr_list, Tr)

    x = [150, 2500]
    y = [Tav_, Tav_]
    plt.plot(x, y, c="r")
    plt.plot(Vrange, Tr_list)

    plt.legend(["Tavailable", "Trequired"])
    plt.title("Σύγκριση της διαθέσιμης με την απαιτούμενη ώση")
    plt.ylabel("Ώση σε lbf")
    plt.xlabel("Ταχύτητα σε ft/s")
    Vmax = (((Tav_ / W) * (W / S) + (W / S) * ((((Tav_ / W) ** 2) - (4 * Cd0 * K)) ** (1 / 2))) / (rho * Cd0)) ** (
                1 / 2)
    print("Η αναλυτική έκφραση της μέγιστης ταχύτητας για το υψόμετρο = " + str(altitude) +
          " δίνει: " + str(round(Vmax, 2)) + " ft/s")
    return Tr_list


def Prequired():
    Tr_list = Trequired(0)
    Pr_list = np.array([])
    Pa_list = np.array([])
    plt.clf()
    j = 0
    rho = 2.3769 * (10 ** (-3))
    max_ROC = 0
    Vrange = np.arange(150, 1500, 50)

    ### 5o erwthma ####
    for V in Vrange:
        Pr = (V * Tr_list[j])
        Pa = (V * Tav)
        diff = Pa - Pr
        if diff > max_ROC:
            max_ROC = diff
            V_rc = V
        Pa_list = np.append(Pa_list, Pa)
        Pr_list = np.append(Pr_list, Pr)
        j = j + 1

    max_ROC = (max_ROC / W) * 60

    print("Το μέγιστο Rate of Climb στο επίπεδο της θάλλασας είναι: " + str(round(max_ROC, 2)) + " ft/min")
    print("ΤΗ ταχύτητα στην οποία έχουμε μέγιστο ρυθμό αναρρίχησης είναι: " + str(round(V_rc, 2)) + " ft/s")
    print("\n")

    plt.plot(Vrange, Pa_list)
    plt.plot(Vrange, Pr_list)
    plt.ylabel("Ισχύς σε ft*lb/s")
    plt.xlabel("Ταχύτητα σε ft/s")
    plt.legend(["Pavailable", "Prequired"])


def climb():
    rho_0 = 2.3769 * (10 ** (-3))
    Tav_ = Tclimb
    ### 6o erwthma ####
    Z = 1 + ((1 + (3 / ((Cl_Cd_max ** 2) * ((Tav_ / W) ** 2)))) ** (1 / 2))
    R_c_max = ((((W / S) * Z) / (3 * rho_0 * Cd0)) ** (1 / 2)) * ((Tav_ / W) ** (3 / 2)) * (
                1 - (Z / 6) - (3 / (2 * ((Tav_ / W) ** 2) * (Cl_Cd_max ** 2) * Z)))
    V_rc_max = (((Tav_ / W) * (W / S) * Z) / (3 * rho_0 * Cd0)) ** (1 / 2)

    #### 7o erwthma ####
    sinth_max = (Tav_ / W) - ((4 * Cd0 * K) ** (1 / 2))
    theta_max = degrees(asin(sinth_max))
    V_thetamax = ((2 / rho_0) * ((K / Cd0) ** (1 / 2)) * (W / S) * cos(radians(theta_max))) ** (1 / 2)

    #### 9o erwthma ###
    "pinakas me puknotes ana 2000ft apo 0ft ws 48000ft"
    density = np.array(
        [2.3768, 2.2409, 2.1110, 1.9869, 1.8685, 1.7556, 1.6480, 1.5455, 1.4480, 1.3553, 1.2673, 1.1836, 1.1043, 1.0292,
         0.95801, 0.89068, 0.82704, 0.76696, 0.71028, 0.64629, 0.58727, 0.53365, 0.48493, 0.44067])
    density = density / 1000  # slug/ft^3
    R_c_max_2 = np.array([])
    for rho_new in density:
        T = Tav_ * ((rho_new / 0.002377) ** (0.9))
        Z = 1 + ((1 + (3 / ((Cl_Cd_max ** 2) * ((T / W) ** 2)))) ** (1 / 2))
        R_c_max_new = ((((W / S) * Z) / (3 * rho_new * Cd0)) ** (1 / 2)) * ((T / W) ** (3 / 2)) * (
                1 - (Z / 6) - (3 / (2 * ((T / W) ** 2) * (Cl_Cd_max ** 2) * Z)))
        R_c_max_2 = np.append(R_c_max_2, R_c_max_new)

    plt.plot(R_c_max_2, np.arange(0, 48000, 2000))
    plt.title("Μέγιστο R/C σε συνάρτηση με το υψόμετρο")
    plt.ylabel("Υψόμετρο ft")
    plt.xlabel("Μέγιστος ρυθμός αναρρίχησης ft/s")

    print("############### 6o erwthma ###################")
    print(
        "Με τη αναλυτική σχέση για υψόμετρο = 0 έχω μέγιστο Rate of Climb: " + str(round(R_c_max * 60, 2)) + " ft/min")
    print("Η ταχύτητα κατά το μέγιστο ρυθμό αναρρίχησης με την αναλυτική σχέση είναι: " + str(
        round(V_rc_max, 2)) + " ft/s")
    print("\n")
    print("############### 7o erwthma ###################")
    print("Η μέγιστη γωνία αναρρίχησης είναι " + str(round(theta_max, 2)) + " μοίρες")
    print("Η ταχύτητα στη μέγιστη γωνία αναρρίχησης είναι " + str(round(V_thetamax, 2)) + " ft/s")
    print("\n")
    print("############### 9o erwthma ###################")
    print("Το υπηρεσιακό επίπεδο 'οροφής' όπου ο ρυθμός αναρρίχησης είναι 1.67ft/s είναι τα 45000 ft ")


def glide():
    #### 8o erwthma ###
    rho_30 = 8.9068 * (10 ** (-4))  # sta 30000 podia gia to glide
    Tanth_min = 1 / (Cl_Cd_max)
    R_max = 30000 / (Tanth_min)
    th_min_gld = degrees(atan(Tanth_min))
    V_glide = ((2 / rho_30) * ((K / Cd0) ** (1 / 2)) * (W / S)) ** (1 / 2)
    "Na to dw"  #####################
    Vv_min = ((2 / rho_30) * (((K / (3 * Cd0)) * (W / S)) ** (1 / 2))) ** (1 / 2)  # NA TO DW AUTO.
    Vv_real = V_glide * sin(radians(th_min_gld))

    print("############### 8o erwthma ###################")
    print("Η ελάχιστη γωνία θ κατά το glide είναι " + str(round(th_min_gld, 2)) + " μοίρες")
    print("Η μέγιστη απόσταση που μπορεί να φθάσει από τα 30000 είναι: " + str(round(R_max / 5280, 2)) + " miles")
    print("Η ταχύτητα πτήσης για τη μινιμουν γωνία glide είναι για (L/D)max, δηλαδη : " + str(
        round(V_glide, 2)) + " ft/s")
    print("Η μίνιμουμ ταχύτητα 'πτώσης/βυθίσματος' είναι: " + str(
        round(Vv_min, 2)) + " ft/s" + " ενώ η πραγματική είναι: " + str(round(Vv_real, 2)) + " ft/s")


def range():
    #### 10o erwthma ###
    rho_30 = 8.9068 * (10 ** (-4))  # sta 30000 podia gia to glide
    W1 = W - Wf
    ct_s = ct / 3600  ## ana sec
    Range = (2 / ct_s) * ((2 / (rho_30 * S)) ** (1 / 2)) * (Cl_1_2_Cd_max) * ((W ** 0.5) - (W1 ** 0.5))
    Range = Range / 5280  # se milia

    print("############### 10o erwthma ###################")
    print("Η απόσταση (μέγιστη) που μπορεί να διανύσει το αεροσκάφος στα 30000 πόδια είναι: " + str(
        round(Range, 2)) + " miles")


def takeoff():
    rho_0 = 2.3769 * (10 ** (-3))
    # double slotted flaps and slats
    mr = 0.04  # suntelesths trivhs, dry asphalt, brakes off
    Kuc = 4.5 * (10 ** (-5))  # moderate flaps
    h = 12  # ft to upsos twn fterwn se MTW apo boeing
    Cl = 0.1  # sto ground roll
    g = 32.2  # varuthta se ft/s^2
    N = 3  # deuterolepta gia to rotate
    hOB = 50  # ft to upsos tou empodiou
    #####
    Cl_max = cos(radians(Lamda)) * 2.0  # 2.5 apo pinaka 5.3 gia moderate extension
    Vstall = ((2 / rho_0) * (W / S) * (1 / Cl_max)) ** (1 / 2)
    V_LO = 1.15 * Vstall
    T = Tav  # sto takeoff xanw logw speed to 10% to thrust
    Kt = (T / W) - mr
    W_S = (W / S) * (4.448 / 1) * ((1 / 0.3048) ** 2)  # To W/S se N/m2
    mass = W * (0.4536 / 1)  # maza se kila
    D_Cd0 = W_S * Kuc * (mass ** (-0.215))
    G = (((16 * h) / b) ** 2) / (1 + (((16 * h) / b) ** (2)))  # gia to ground effect
    Ka = -((rho_0) / (2 * (W / S))) * (Cd0 + D_Cd0 + ((k1 + (G / (3.14 * e_span * AR))) * (Cl ** 2)) - (mr * Cl))
    #### to Sg finally ##
    sg = ((1 / (2 * g * Ka)) * np.log(1 + (Ka / Kt) * (V_LO ** 2))) + N * V_LO
    print("Η απόσταση του ground roll είναι ίση με: " + str(round(sg, 2)) + " ft")
    ### παμε για το sa
    R = (6.96 * (Vstall ** 2)) / g
    theta_OB = degrees(acos(1 - (hOB / R)))
    sa = R * sin(radians(theta_OB))
    print("Η απόσταση στον αέρα για αποφυγή του εμποδίου είναι ίση με: " + str(round(sa, 2)) + " ft")
    print("Η συνολική απόσταση για απογείωση είναι: " + str(round(sa + sg, 2)) + " ft")


def landing():
    rho_0 = 2.3769 * (10 ** (-3))
    g = 32.2  # ft/s^2
    theta_a = 2.5  # 2.5 moires approach gwnia
    mr = 0.3  # dry asphalt with brakes on
    Trev = 0
    h = 12  # ft apto edafos ta ftera
    Kuc = 3.16 * (10 ** (-5))  # gia megista flaps deflection
    Cl = 0.1  # gia approach
    N = 3  # gia na ginei to free roll
    W = 146000  # max landing weight
    #####################
    Cl_max = cos(radians(Lamda)) * 2.6  # 3.2 apo pinaka 5.3 gia full extension gia landing
    Vstall = ((2 / rho_0) * (W / S) * (1 / Cl_max)) ** (1 / 2)
    Vflare = 1.23 * Vstall
    Vtd = 1.15 * Vstall
    R = (Vflare ** 2) / (0.2 * g)
    hf = R * (1 - cos(radians(theta_a)))
    sa = (50 - hf) / (tan(radians(theta_a)))
    sf = R * sin(radians(theta_a))
    Jt = (Trev / W) + mr
    G = (((16 * h) / b) ** 2) / (1 + (((16 * h) / b) ** (2)))  # gia to ground effect
    W_S = (W / S) * (4.448 / 1) * ((1 / 0.3048) ** 2)  # To W/S se N/m2
    mass = W * (0.4536 / 1)  # maza se kila
    D_Cd0 = W_S * Kuc * (mass ** (-0.215))
    Ja = ((rho_0) / (2 * (W / S))) * (Cd0 + D_Cd0 + ((k1 + (G * k3)) * (Cl ** 2)) - (mr * Cl))
    sg = N * Vtd + (1 / (2 * g * Ja)) * (np.log(1 + ((Ja / Jt) * (Vtd ** 2))))

    print("Η συνολική απόσταση μαζί με το approach,flare και το ground roll είναι: " + str(
        round(sa + sf + sg, 2)) + " ft")


def new_aircraft():
    #### 10o erwthma ###
    rho_30 = 8.9068 * (10 ** (-4))  # sta 30000 podia gia to glide
    ct = 0.53
    S = 1370  # ft^2 apo wikipedia
    ct_s = ct / 3600  ## ana sec
    Range = 3481.37 * 5280  # se ft einai idia h emveleia
    var = (2 / ct_s) * ((2 / (rho_30 * S)) ** (1 / 2)) * (
        Cl_1_2_Cd_max)  # για ευκολια πραξεων ειναι οτι προηγειται του (Wo - W1) στη εξισωση του range
    W1_12 = (W ** (1 / 2)) - (Range / var)
    Wf = W - ((W1_12) ** 2)

    print("Το καύσιμο που καταναλώνουμε για την ίδια εμβέλεια είναι: " + str(round(Wf, 2)) + " lb")


"""Για την απαιτουμενη ωση απλα αλλαζουμε το υψομετρο 0/10000/30000 ποδια.
Τα υπολοιπα τρεχουν οπως ειναι οι συναρτησεις χωρις εισοδους."""

# Trequired(30000)
# Prequired()
# climb()
# glide()
# range()
# takeoff()
landing()
# new_aircraft()

"""απαιτειται για να μενουν τα διαγραμματα"""
plt.show()
