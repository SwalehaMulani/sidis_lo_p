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
parser.add_argument("--zh", type=float, required=True, help="Longitudinal momentum fraction of hadron")

args = parser.parse_args()

#Parameters

Q2 = args.Q2
xbj = args.xbj
zh = args.zh  
a_array = [0.5,1,2]
print("Running gamma-proton T scattering with Q2 =", Q2, ", xbj =", xbj, ", zh =", zh)

############################################################################################
Nc = 3
aem = 1/137
mq = 0.14 #GeV light quark mass

data = pd.read_csv("/users/swanismu/projects/sidislo/gamma_proton/x_g_scale/Qs2_red_curve_10pts.csv")

#sigma0 = 2 * 14.343486992997 #mb from carlisle fit data for MVe model
sigma0 = 2 * 18.81 #sigma0 taken from MV fit parameters of table I of 1309.6963 #mb from carlisle fit data
P_vec = np.linspace(1,15,57) #GeV

##############______________________________________####################################

#define th functions required for the integrand of xsection
#bk_file = os.path.join("/users/swanismu/projects/sidislo/gamma_proton/BK_momentum_space", "BKfit_Sk_data.csv") #for MVe model
bk_file = os.path.join("/users/swanismu/projects/sidislo/MV_bk_momentum_space", "2Dft_dipole.csv") #for MV model fitted data from  with initial condition (arXiv: 1309.6963)
interp = dipoleMomNew.BKDipoleMomentum(bk_file)
#define dipole amplitude in momentum space
#def dip(k):
    #N =  2 * np.pi * interp.S_dipole(yevol, k)  #fixed rapidity y=2
    #return N

def dip(p, k, z1):

    valid_z1 = (z1 > 0) & (z1 < 1)

    #z1_safe = np.where(valid_z1, z1, 0.5)

    Q2bar = z1 * (1 - z1) * Q2

    Qs2 = np.interp(
        xbj,
        data["xBj"],
        data["Qs2_GeV2"]
    )

    xg = xbj * (
        1
        + p**2 / (z1 * Q2)
        + np.maximum(Q2bar, Qs2)
          / ((1 - z1) * Q2)
    )

    # Freeze xg at 0.01
    xg_eval = np.minimum(xg, 0.01)

    # Valid points for logarithm
    mask = (
        valid_z1
        & np.isfinite(xg_eval)
        & (xg_eval > 0)
    )

    Np = np.zeros_like(xg, dtype=float)

    if np.any(mask):

        indices = np.flatnonzero(mask)

        yevol = np.log(
            0.01 / xg_eval.flat[indices]
        )

        mask_y = (
            np.isfinite(yevol)
            & (yevol <= 30.0)
        )

        if np.any(mask_y):

            valid_indices = indices[mask_y]

            Np.flat[valid_indices] = (
                2 * np.pi *
                interp.S_dipole(
                    yevol[mask_y],
                    k.flat[valid_indices]
                )
            )

    return Np
    
#print (dip(100))

##############______________________________________####################################

#fixing fragmentation function D(zf)

#import scipy.integrate as integrate
ff = lhapdf.getPDFSet("NNFF10_PIsum_lo").mkPDF(0)

#Q = np.sqrt(Q2) # energy scale in GeV

def D_light(Q, zf):
    """Integrate and sum the three light flavors (u, d, s)."""
    flavors = {
        2: (2/3)**2,   # u
        1: (1/3)**2,   # d
        3: (1/3)**2,   # s
    }

    zf = np.asarray(zf)
    scalar_input = zf.ndim == 0
    zf = np.atleast_1d(zf)

    result = np.zeros_like(zf, dtype=float)
    valid = (zf > 0.0) & (zf < 1.0)

    for pid, ef2 in flavors.items():
        ff_vals = np.array([
            ff.xfxQ(pid, z, Q) / z if v else 0.0
            for z, v in zip(zf, valid)
        ])
        result += ef2 * ff_vals

    return result[0] if scalar_input else result


###############______________________________________####################################

#function to save results to csv file

def save_results(Pper, mean_T, sdev_T, intial_time, final_time):

    # Define the output folder path
    #output_folder = os.path.join("/users/swanismu/projects/sidislo/gamma_proton/xsection_in_mb/new_z1/MV_fit/106_points_P_Pper_15", f"output_EIC_Q2{Q2}_xbj{xbj}")
    output_folder = os.path.join("/users/swanismu/projects/sidislo/gamma_proton/x_g_scale/MV_fit/EIC_106_points_P_Pper_15_new", f"output_EIC_Q2{Q2}_xbj{xbj}")


    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create dynamic filename
    filename = os.path.join(output_folder, f"gammaP_T_Q2{Q2}_xbj{xbj}_zh{zh}_mb_MV_{a}.csv")

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
@vegas.rbatchintegrand
def SIDIS_integrand(kper, zf):
       z1 = zh / zf
       valid = (z1 > 0) & (z1 < 1)
       z2 = 1 - z1 #z2 is fixed by z1 only for this code for simplicity
       arg = Q2 * z1 * z2 + mq**2
       eps = np.sqrt(np.where(valid, arg, 0.0))
       H =(aem * Nc * sigma0 ) / (2*(2*np.pi)**4)
       Pquark = Pper/zf
       M1 = kper *(-eps**2/(Pquark**2 + eps**2)**2 + (kper**2 + 2 * eps**2) /((Pquark**2 + eps**2) * ((eps**2 + Pquark**2 + kper**2)**2 - 4 * kper**2 * Pquark**2)**(1/2)) -  (eps**2 * (eps**2 + Pquark**2 + kper**2))/((eps**2 + Pquark**2 + kper**2)**2 - 4 * kper**2 * Pquark**2)**(3/2))
       M2 = kper *(1/(eps**2 + Pquark**2)**2 - ((eps**2 + Pquark**2 + kper**2)*(eps**2 + Pquark**2 + 2 * kper**2) - 8 * kper**2 * Pquark**2)/((eps**2 + Pquark**2)*((eps**2 + Pquark**2 + kper**2)**2 - 4 * kper**2 * Pquark**2)**(3/2)))
       Z = 2*(z1**2 + (1-z1)**2)
       if (a*Pper <= 1.0):
            D_light_sum = 2 * D_light(1, zf)
       else:
            D_light_sum = 2 * D_light(a*Pper, zf)
       result = H *( Z * M1 + mq**2 * M2 )* dip(Pquark, kper, z1) *D_light_sum * (1/zf**3)
       result = np.where(valid, result, 0.0)
       return  result # multiplying with Pper to get d sigma / d Pper format similar to HERA data

@vegas.rbatchintegrand   
def integrand(x):
        kper, zf = x
        return SIDIS_integrand(kper, zf)
    
lims = [[0.01, 100], [0.01, 1]] #integration limits for kper and zf 

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
        integ(integrand, nitn=15, neval = 1e6, alpha = 0.5, nproc = 36) #warmup

        #initialize variables for mean and sdev
        mean_T, sdev_T = 1e-15, 1e-15

        result_T = integ(integrand, nitn=50, neval = 1e6, alpha = 0.5, nproc = 36)
        mean_T, sdev_T = result_T.mean, result_T.sdev
        
        # store results
        mean_T_list.append(mean_T)
        sdev_T_list.append(sdev_T)

        final_time = time.time()

        print("Pper = ", Pper, "mean_T = ", mean_T, "+/-", sdev_T, "Time taken (s): ", final_time - intial_time)
        # Write a new row of data
        save_results(Pper, mean_T, sdev_T, intial_time, final_time)