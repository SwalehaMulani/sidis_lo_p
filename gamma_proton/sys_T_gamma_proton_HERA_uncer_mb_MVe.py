import numpy as np
import vegas
import lhapdf
import time
import os
import csv
import dipoleMomNew # Import the dipole momentum module
import argparse
import pandas as pd


#######_______________________________________####################################
# User-defined parameters
############################################################################################
parser = argparse.ArgumentParser(description="SIDIS parameters")

parser.add_argument("--Q2", type=float, required=True, help="Q^2 in GeV^2")
parser.add_argument("--xbj", type=float, required=True, help="Bjorken x")
parser.add_argument("--y", type=float, required=True, help="rapidity")

args = parser.parse_args()

#Parameters

Q2 = args.Q2
xbj = args.xbj
y = args.y  #kinematical rapidity talen from HERA data, independant variable

############################################################################################
Nc = 3
aem = 1/137
mq = 0.14 #GeV light quark mass

#parameters from F2 table1 HERA data 9610006
W2 = Q2 / xbj #GeV^2 , this is approx formula neglecting proton mass
#yevol = np.log(0.01/xbj) #evolution rapidity

sigma0 = 2 * 14.343486992997 #mb from carlisle fit data for MVe Model
#sigma0 =  2 * 18.81 #sigma0 taken from MV fit parameters of table I of 1309.6963
data = pd.read_csv("/users/swanismu/projects/sidislo/gamma_proton/x_g_scale/Qs2_red_curve_10pts.csv")


P_vec = [1.1, 1.3, 1.5,1.7,1.95,2.3, 2.75, 3.5,4.5] #GeV
a_array = [0.5,1,2]
print("Running gamma-proton T scattering with Q2 =", Q2, ", xbj =", xbj, ", y =", y, "uncertainty scale a =", a_array)

##############______________________________________####################################

#define th functions required for the integrand of xsection
#bk_file = os.path.join("/users/swanismu/projects/sidislo/gamma_proton/BK_momentum_space", "BKfit_Sk_data.csv")
bk_file = os.path.join("/users/swanismu/projects/sidislo/MVe_BK_proton_Qsr_fit/Sk_fit/10e-3min", "2Dft_0.csv") #for MVe model Qsr fit with 10-3 min
#bk_file = os.path.join("/users/swanismu/projects/sidislo/gamma_proton/xsection_in_mb/MV_bk_momentum_space", "2Dft_dipole.csv") #MV
interp = dipoleMomNew.BKDipoleMomentum(bk_file)
#define dipole amplitude in momentum space
#def dip(k):
    #N =  2* np.pi * interp.S_dipole(yevol, k)  #fixed rapidity y=2
    #return N
    
def dip(p, k, z1):

    # Check physical z1 range
    if not (0 < z1 < 1):
        return 0.0

    # Calculate Qbar^2
    Q2bar = z1 * (1 - z1) * Q2

    # Saturation scale
    Qs2 = np.interp(
        xbj,
        data["xBj"],
        data["Qs2_GeV2"]
    )

    # Calculate xg
    xg = xbj * (
        1
        + p**2 / (z1 * Q2)
        + np.maximum(Q2bar, Qs2) / ((1 - z1) * Q2)
    )

    # Freeze xg at 0.01
    # xg < 0.01  -> actual xg
    # xg >= 0.01 -> xg = 0.01
    xg_eval = min(xg, 0.01)

    # Evolution variable
    yevol = np.log(0.01 / xg_eval)

    # Check interpolation range
    if yevol > 30.0:
        return 0.0

    # Evaluate proton dipole
    Np = (
        2 * np.pi *
        interp.S_dipole(
            yevol,
            k
        )
    )

    return Np

#print (dip(100))

##############______________________________________####################################

#fixing fragmentation function D(zf)

#import scipy.integrate as integrate
ff = lhapdf.getPDFSet("NNFF10_PIsum_lo").mkPDF(0)
ffK = lhapdf.getPDFSet("NNFF10_KAsum_lo").mkPDF(0)
ffPR = lhapdf.getPDFSet("NNFF10_PRsum_lo").mkPDF(0)

FLAVORS = {
    2: (2/3)**2,   # u
    1: (1/3)**2,   # d
    3: (1/3)**2,   # s
}

def D_light(Q, zf, ff_set):
    """
    Charge-weighted sum of D_{q-h}(zf, Q) over light flavors u, d, s
    for a given fragmentation function set ff_set.
    Factor 2 accounts for antiquark contributions.

    Parameters
    ----------
    Q      : float  - factorisation scale [GeV]
    zf     : float  - fragmentation variable
    ff_set : lhapdf PDF object  - ff, ffK, or ffPR
    """
    if zf <= 0.0 or zf >= 1.0:
        return 0.0

    total_sum = 0.0
    for pid, ef2 in FLAVORS.items():
        total_sum += ef2 * ff_set.xfxQ(pid, zf, Q) / zf
    return total_sum   


def D_pi(Q, zf):
    """Pion fragmentation function."""
    return D_light(Q, zf, ff)

def D_kaon(Q, zf):
    """Kaon fragmentation function."""
    return D_light(Q, zf, ffK)

def D_proton(Q, zf):
    """Proton fragmentation function."""
    return D_light(Q, zf, ffPR)

def D_total(Q, zf):
    """
    Sum of all hadron fragmentation functions:
    pion + kaon + proton.
    Use this in the integrand to get the inclusive charged hadron cross section.
    """
    return D_pi(Q, zf) + D_kaon(Q, zf) + D_proton(Q, zf)
###############______________________________________####################################

#function to save results to csv file

def save_results(Pper, mean_T, sdev_T, intial_time, final_time):

    # Define the output folder path
    output_folder = os.path.join("/users/swanismu/projects/sidislo/gamma_proton/x_g_scale/MVe_fit", f"output_HERA_uncertainty_hadron_pq")


    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create dynamic filename
    filename = os.path.join(output_folder, f"gammaP_Pper_scale_T_Q2{Q2}_xbj{xbj}_y{y}_mb_MVe_{a}.csv")

    # Check if file exists
    file_exists = os.path.isfile(filename)

    # Open in append mode
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
    
        # If file is new, write header first
        if not file_exists:
            writer.writerow(["Pper", "mean_T", "sdev_T", "time_taken_sec"])
    
        writer.writerow([Pper, mean_T, sdev_T, final_time - intial_time])

###############______________________________________####################################
#integrand function for transverse cross section

def SIDIS_integrand(kper, zf):
       z1 = np.exp(y) * (np.sqrt(Pper**2 + mq**2)) / ( zf * np.sqrt(W2)) #longitudinal momentum fraction of quark, check formula for mT = sqrt(zf2 * pper2 + mq2)
       z2 = 1 - z1 #z2 is fixed by z1 only for this code for simplicity
       if (z1 <=0) or (z1 >=1):
        return 0
       eps = np.sqrt(Q2 * z1 * z2 + mq**2)
       H =(aem * Nc * sigma0 ) / (2*(2*np.pi)**4)
       Pquark = Pper/zf
       M1 = kper *(-eps**2/(Pquark**2 + eps**2)**2 + (kper**2 + 2 * eps**2) /((Pquark**2 + eps**2) * ((eps**2 + Pquark**2 + kper**2)**2 - 4 * kper**2 * Pquark**2)**(1/2)) -  (eps**2 * (eps**2 + Pquark**2 + kper**2))/((eps**2 + Pquark**2 + kper**2)**2 - 4 * kper**2 * Pquark**2)**(3/2))
       M2 = kper *(1/(eps**2 + Pquark**2)**2 - ((eps**2 + Pquark**2 + kper**2)*(eps**2 + Pquark**2 + 2 * kper**2) - 8 * kper**2 * Pquark**2)/((eps**2 + Pquark**2)*((eps**2 + Pquark**2 + kper**2)**2 - 4 * kper**2 * Pquark**2)**(3/2)))
       Z = 2*(z1**2 + (1-z1)**2)
       if (a*Pper <= 1.0):
          D_light_sum = 2 * D_total(1, zf)
       else:
          D_light_sum = 2 * D_total((a*Pper), zf) #multiply by 2 for antiquark contribution for each flavor of that of quark and 2Pper for uncertainty scale computation
       return H *( Z * M1 + mq**2 * M2 )* dip(Pquark, kper, z1) *D_light_sum * Pper * (1/zf**2) * z1 # multiplying with Pper to get d sigma / d Pper format similar to HERA data and multiplied by z1 from new der (26.07.26)

def integrand(x):
        kper = float(x[0])
        zf = float(x[1])
        val = SIDIS_integrand(kper, zf)
        return float(val)
    
lims = [[0.01, 100], [0.019, 1]] #integration limits for kper and zf 

################______________________________________####################################

# --- results containers ---
mean_T_list = []
sdev_T_list = []

#####################################################################################
# vegas integration
for a in a_array:
    for Pper in P_vec:

        intial_time = time.time()
        #create vega integration object
        integ = vegas.Integrator(lims)

        #perform integration for longitudinal xsection
        integ(integrand, nitn=15, neval = 1e6, alpha = 0.5, nproc = 24) #warmup

        #initialize variables for mean and sdev
        mean_T, sdev_T = 1e-15, 1e-15

        result_T = integ(integrand, nitn=50, neval = 1e6, alpha = 0.5, nproc = 24)
        mean_T, sdev_T = result_T.mean, result_T.sdev

        # store results
        mean_T_list.append(mean_T)
        sdev_T_list.append(sdev_T)

        final_time = time.time()

        print("Pper = ", Pper, "mean_T = ", mean_T, "+/-", sdev_T, "Time taken (s): ", final_time - intial_time)
        # Write a new row of data
        save_results(Pper, mean_T, sdev_T, intial_time, final_time)