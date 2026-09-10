import numpy as np
import vegas
import time
import os
import csv
import dipoleMomNew # Import the dipole momentum module
import lhapdf
import argparse
import pandas as pd

############____________________________###############################

parser = argparse.ArgumentParser(description="SIDIS parameters")

parser.add_argument("--Q2", type=float, required=True, help="Q^2 in GeV^2")
parser.add_argument("--xbj", type=float, required=True, help="Bjorken x")
parser.add_argument("--zh", type=float, required=True, help="Longitudinal momentum fraction of hadron")

args = parser.parse_args()

#parameters
Nc = 3
aem = 1/137
mq = 0.14 #GeV light quark mass

#parameters from F2 table8 HERA data 9610006
Q2 = args.Q2
xbj = args.xbj
zh =  args.zh #fixed to obtain kinametical rapidity
print("Running gamma-proton L EIC with Q2=", Q2, "xbj=", xbj, "zh=", zh)
data = pd.read_csv("/users/swanismu/projects/sidislo/gamma_proton/x_g_scale/Qs2_red_curve_10pts.csv")

#sigma0 = 2 * 14.343486992997 #mb from carlisle fit data for MVe model
sigma0 =  2 * 18.81 #sigma0 taken from MV fit parameters of table I of 1309.6963

P_vec = np.linspace(1,15,57) #GeV
a_array = [0.5,1,2] #for checking FF scale dependence

############____________________________###############################

#define th functions required for the integrand of xsection
#bk_file = os.path.join("/users/swanismu/projects/sidislo/gamma_proton/BK_momentum_space", "BKfit_Sk_data.csv") #for MVe model
bk_file = os.path.join("/users/swanismu/projects/sidislo/MV_bk_momentum_space", "2Dft_dipole.csv") #for MV model fitted data from  with initial condition (arXiv: 1309.6963)
interp = dipoleMomNew.BKDipoleMomentum(bk_file)
#define dipole amplitude in momentum space
#def dip(k):
    #N =  2 * np.pi * interp.S_dipole(yevol, k)  #fixed kinametic evolution rapidity 
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

##########____________________________###############################

#fixing fragmentation function D(zf)

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

##########____________________________###############################

#function to save results to csv file

def save_results(Pper, mean_L, sdev_L, intial_time, final_time):

    # Define the output folder path
    output_folder = os.path.join("/users/swanismu/projects/sidislo/gamma_proton/x_g_scale/MV_fit/EIC_106_points_P_Pper_15_new", f"output_EIC_Q2{Q2}_xbj{xbj}")

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create dynamic filename
    filename = os.path.join(output_folder, f"gammaP_L_Q2{Q2}_xbj{xbj}_zh{zh}_mb_MV_{a}.csv")

    # Check if file exists
    file_exists = os.path.isfile(filename)

    # Open in append mode
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
    
        # If file is new, write header first
        if not file_exists:
            writer.writerow(["Pper", "mean_L", "sdev_L", "time_taken_sec"])
    
        writer.writerow([Pper, mean_L, sdev_L, final_time - intial_time])

##########____________________________###############################
##########___________________________######################################
#Main SIDIS calculation loop over Pper values

for a in a_array:
    for Pper in P_vec:

        #define xsection integrand
        @vegas.rbatchintegrand
        def SIDIS_integrand(kper, zf):
            z1 = zh / zf
            valid = (z1 > 0) & (z1 < 1)
            z2 = 1-z1
            arg = Q2 * z1 * z2 + mq**2
            eps = np.sqrt(np.where(valid, arg, 0.0))
            H =(aem * Nc * sigma0 ) / (2*(2*np.pi)**4)
            Pquark = Pper/zf
            M = kper *(1/(eps**2 + Pquark**2)**2 - ((eps**2 + Pquark**2 + kper**2)*(eps**2 + Pquark**2 + 2 * kper**2) - 8 * kper**2 * Pquark**2)/((eps**2 + Pquark**2)*((eps**2 + Pquark**2 + kper**2)**2 - 4 * kper**2 * Pquark**2)**(3/2)))
            Z = z1*(1-z1)* eps**2 * 8 #z1 here is longititudinal momentum fraction of quark z1= k1+/(k1+ + k2+)
            if (a*Pper <= 1.0):
                D_light_sum = 2 * D_light(1, zf)
            else:
                D_light_sum = 2 * D_light(a*Pper, zf)
            dipole = dip(Pquark, kper, z1)
            result = H * Z * M * dipole*D_light_sum *(1/zf**3) # multiplying with Pper to get d sigma / d Pper format similar to HERA data
            result = np.where(valid, result, 0.0)
            return result
        
        @vegas.rbatchintegrand
        def integrand(x):
            kper, zf = x

            return SIDIS_integrand(kper, zf)
        
        #integration limits
            
        lims = [[0.01, 100], [0.01,1]] #integration limits for kper and zf

        intial_time = time.time()

        #create vegas integration object
        integ = vegas.Integrator(lims)
        
        #perform integration for longitudinal xsection
        integ(integrand, nitn=15, neval = 1e6, alpha = 0.5, nproc = 36) #warmup
        
        #initialize variables for mean and sdev
        mean_L, sdev_L = 1e-15, 1e-15

        result_L = integ(integrand, nitn=50, neval = 1e6, alpha = 0.5, nproc = 36)
        mean_L, sdev_L = result_L.mean, result_L.sdev
    
        final_time = time.time()    
        

        print("Pper = ", Pper, "mean_L = ", mean_L, "+/-", sdev_L, "Time taken:", final_time - intial_time, "seconds")

        #save results to csv file
        print("saving results to.......")
        save_results(Pper, mean_L, sdev_L, intial_time, final_time)
    