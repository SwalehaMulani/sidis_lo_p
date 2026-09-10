import numpy as np
import vegas
import time
import os
import csv
import dipoleMomNew # Import the dipole momentum b independent module
import dipoleMom_b_dep # Import the dipole momentum b dependent module
import ws_interpolator #Woods-Saxon function calculator class
import lhapdf
import argparse
import pandas as pd

############___________________________________#################################

parser = argparse.ArgumentParser(description="SIDIS parameters")
parser.add_argument("--sample", type=int, required=True, help="Posterior sample index")

args = parser.parse_args()

#parameters
Nc = 3
aem = 1/137
mq = 0.14 #GeV light quark mass
A = 197 #Au nucleus

zh = 0.5 
Q2_vec = [5, 10, 50] #GeV^2
xbj_vec = [0.01, 0.008, 0.006, 0.004, 0.002, 0.001, 0.0008, 0.0006, 0.0004, 0.0002, 0.0001] #Bjorken x values to be taken from EIC data kinematic bins

RA = (1.12 * A**(1/3)  - 0.86 * A**(-1/3))*5.06773 #GeV-1 nuclear radius
d = 0.54*5.06773 #Gev^-1 0.54fm skin depth
ws_interpolator_obj = ws_interpolator.WoodsSaxonInterpolator(A, RA, d)
WS_TA_b = ws_interpolator_obj.integrated_TA_b() # Woods-Saxon nuclear thickness function T_A(b) value
TA_b0 = ws_interpolator_obj.TA_b(0)
print("TA_b0", TA_b0)
data = pd.read_csv("/users/swanismu/projects/sidislo/gamma_nucleus/x_g_scale/Qs2_red_curve_10pts.csv")

P_vec = np.linspace(1,15,57) #GeV


#####################______________________________###########################
sample_id = args.sample

#option1 : extract sigma0_2 from the posterior parameter file for the sample index
posterior_file = "/users/swanismu/projects/sidislo/bk_momentum_posterior_10-3_P/posteriorsamples_100_LOmvefit.dat"
posterior_data = np.loadtxt(posterior_file, comments="#") #load the posterior samples from the file and skip any comment lines starting with '#'
sigma0_2 = posterior_data[sample_id, 3]
sigma0 = 2.0 * sigma0_2 * 2.56819 #convert from mb to GeV^-2
#sigma0 = 2.0 * sigma0_2  #mb 

#load BK dipole fit data for proton
bk_dir_proton = "/users/swanismu/projects/sidislo/bk_momentum_posterior_10-3_P/10e-3min"
bk_file_proton = os.path.join(bk_dir_proton, f"2Dft_{sample_id}.csv")
interp_proton = dipoleMomNew.BKDipoleMomentum(bk_file_proton)

#define dipole amplitude in momentum space for proton
#def dip_proton(k):
    #Np =  2 * np.pi * interp_proton.S_dipole(yevol, k)  #fixed rapidity 
    #return Np
def dip_proton(p, k, z1):

    # ---------------------------------------------------------
    # 1. Physical z1 validity
    # ---------------------------------------------------------
    valid_z1 = (z1 > 0) & (z1 < 1)

    # ---------------------------------------------------------
    # 2. Calculate Qbar^2
    # ---------------------------------------------------------
    Q2bar = z1 * (1 - z1) * Q2

    # ---------------------------------------------------------
    # 3. Saturation scale
    # ---------------------------------------------------------
    Qs2 = np.interp(
        xbj,
        data["xBj"],
        data["Qs2_GeV2"]
    )

    # ---------------------------------------------------------
    # 4. Calculate xg
    # ---------------------------------------------------------
    xg = xbj * (
        1
        + p**2 / (z1 * Q2)
        + np.maximum(Q2bar, Qs2)
          / ((1 - z1) * Q2)
    )

    # ---------------------------------------------------------
    # 5. Freeze xg at 0.01
    #    xg < 0.01  -> use actual xg
    #    xg >= 0.01 -> use xg = 0.01
    # ---------------------------------------------------------
    xg_eval = np.minimum(xg, 0.01)

    # ---------------------------------------------------------
    # 6. Valid points for logarithm
    # ---------------------------------------------------------
    mask = (
        valid_z1
        & np.isfinite(xg_eval)
        & (xg_eval > 0)
    )

    # ---------------------------------------------------------
    # 7. Initialize output
    # ---------------------------------------------------------
    Np = np.zeros_like(xg, dtype=float)

    # ---------------------------------------------------------
    # 8. Calculate dipole only for valid points
    # ---------------------------------------------------------
    if np.any(mask):

        indices = np.flatnonzero(mask)

        # Evolution variable
        yevol = np.log(
            0.01 / xg_eval.flat[indices]
        )

        # Check interpolation range
        mask_y = (
            np.isfinite(yevol)
            & (yevol <= 30.0)
        )

        if np.any(mask_y):

            valid_indices = indices[mask_y]

            Np.flat[valid_indices] = (
                2 * np.pi *
                interp_proton.S_dipole(
                    yevol[mask_y],
                    k.flat[valid_indices]
                )
            )

    return Np
    


#load BK dipole fit data for nucleus
bk_dir_nucleus = "/users/swanismu/projects/sidislo/MVe_BK_10e-3min_posterior_A"
bk_file_nucleus = os.path.join(bk_dir_nucleus,  f"2Dft_{sample_id}.csv")
interp_nucleus = dipoleMom_b_dep.BKDipoleMomentum(bk_file_nucleus)

#define dipole amplitude in momentum space
#def dip_nucleus(b, k):
    #Nn =  2 * np.pi * interp_nucleus.S_dipole(b, yevol, k)  #fixed rapidity
    #return Nn
    
def dip_nucleus(p, b, k, z1):

    # ---------------------------------------------------------
    # 1. Physical z1 validity
    # ---------------------------------------------------------
    valid_z1 = (z1 > 0) & (z1 < 1)

    # ---------------------------------------------------------
    # 2. Calculate Qbar^2
    # ---------------------------------------------------------
    Q2bar = z1 * (1 - z1) * Q2

    # ---------------------------------------------------------
    # 3. Saturation scale
    # ---------------------------------------------------------
    Qs2 = (sigma0 * A * TA_b0 * np.interp( xbj, data["xBj"], data["Qs2_GeV2"])) / 2 
    #print("Qs2", Qs2)
    # ---------------------------------------------------------
    # 4. Calculate xg
    # ---------------------------------------------------------
    xg = xbj * (
        1
        + p**2 / (z1 * Q2)
        + np.maximum(Q2bar, Qs2)
          / ((1 - z1) * Q2)
    )

    # ---------------------------------------------------------
    # 5. Freeze xg at 0.01
    #    xg < 0.01  -> use actual xg
    #    xg >= 0.01 -> use xg = 0.01
    # ---------------------------------------------------------
    xg_eval = np.minimum(xg, 0.01)

    # ---------------------------------------------------------
    # 6. Valid points for logarithm
    # ---------------------------------------------------------
    mask = (
        valid_z1
        & np.isfinite(xg_eval)
        & (xg_eval > 0)
    )

    # ---------------------------------------------------------
    # 7. Initialize output
    # ---------------------------------------------------------
    NA = np.zeros_like(xg, dtype=float)

    # ---------------------------------------------------------
    # 8. Calculate dipole only for valid points
    # ---------------------------------------------------------
    if np.any(mask):

        indices = np.flatnonzero(mask)

        # Evolution variable
        yevol = np.log(
            0.01 / xg_eval.flat[indices]
        )

        # Check interpolation range
        mask_y = (
            np.isfinite(yevol)
            & (yevol <= 30.0)
        )

        if np.any(mask_y):

            valid_indices = indices[mask_y]

            NA.flat[valid_indices] = (
                2 * np.pi *
                interp_nucleus.S_dipole(
                    b.flat[valid_indices],
                    yevol[mask_y],
                    k.flat[valid_indices]
                )
            )

    return NA
#####################______________________________###########################

#fixing fragmentation function D(zf)

ff = lhapdf.getPDFSet("NNFF10_PIsum_lo").mkPDF(0)
#ff = lhapdf.getPDFSet("NNFF10_KAsum_lo").mkPDF(0)

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
 ##################______________________________###########################
@vegas.rbatchintegrand
def SIDIS_integrand_1(b, kper, zf):
            z1 = zh / zf #z1 is longitudinal momentum fraction of quark z1= k1+/(k1+ + k2+) fixed by zh and zf
            valid = (z1 > 0) & (z1 < 1)
            z2 = 1 - z1 #z2 is fixed by z1 only for this code for simplicity
            arg = Q2 * z1 * z2 + mq**2
            eps = np.sqrt(np.where(valid, arg, 0.0))
            H =(2* np.pi* aem * Nc ) / (2*np.pi)**4
            Pquark = Pper/zf
            M = kper *(1/(eps**2 + Pquark**2)**2 - ((eps**2 + Pquark**2 + kper**2)*(eps**2 + Pquark**2 + 2 * kper**2) - 8 * kper**2 * Pquark**2)/((eps**2 + Pquark**2)*((eps**2 + Pquark**2 + kper**2)**2 - 4 * kper**2 * Pquark**2)**(3/2)))
            Z = z1*(1-z1)* eps**2 * 8 #z1 here is longititudinal momentum fraction of quark z1= k1+/(k1+ + k2+)
            D_light_sum = 2 * D_light(Pper, zf) #multiply by 2 for antiquark contribution for each flavor of that of quark
            dipole = dip_nucleus(Pquark, b, kper, z1)
            result = H * Z * M * dipole*D_light_sum * b * (1/zf**3) 
            result = np.where(valid, result, 0.0)
            return result

        ##################______________________________###########################
@vegas.rbatchintegrand
def integrand_1(x):
    b, kper, zf = x                
    return SIDIS_integrand_1(b, kper, zf)
            
lims = [[0,30], [0.01, 100], [0.01,1]] #integration limits for b(impact parameter), kper(internal transverse momentum) and zf(fragmentation variable)

#define xsection integrand 2 
#__________________________________________________________________#
@vegas.rbatchintegrand
def SIDIS_integrand_2( kper, zf):
    z1 = zh / zf #z1 is longitudinal momentum fraction of quark z1= k1+/(k1+ + k2+) fixed by zh and zf
    valid = (z1 > 0) & (z1 < 1)
    z2 = 1 - z1 #z2 is fixed by z1 only for this code for simplicity
    arg = Q2 * z1 * z2 + mq**2
    eps = np.sqrt(np.where(valid, arg, 0.0))
    H =(2* np.pi* aem * Nc ) / (2*np.pi)**4

    Pquark = Pper/zf
    M = kper *(1/(eps**2 + Pquark**2)**2 - ((eps**2 + Pquark**2 + kper**2)*(eps**2 + Pquark**2 + 2 * kper**2) - 8 * kper**2 * Pquark**2)/((eps**2 + Pquark**2)*((eps**2 + Pquark**2 + kper**2)**2 - 4 * kper**2 * Pquark**2)**(3/2)))
    Z = z1*(1-z1)* eps**2 * 8 #z1 here is longititudinal momentum fraction of quark z1= k1+/(k1+ + k2+)
    S = (sigma0 * A) / 2
    WS = S *  WS_TA_b # Woods-Saxon nuclear thickness function T_A(b)
    D_light_sum = 2 * D_light(Pper, zf) #multiply by 2 for antiquark contribution for each flavor of that of quark
    dipole = dip_proton(Pquark, kper, z1)
    result = H * Z * M * dipole * D_light_sum * WS * (1/zf**3)  # multiplying with Pper to get d sigma / d Pper format similar to HERA data
    result = np.where(valid, result, 0.0)
    return result
##################______________________________###########################
@vegas.rbatchintegrand
def integrand_2(x):
    kper, zf = x                
    return SIDIS_integrand_2(kper, zf)
            
lims_2 = [[0.01, 100], [0.01,1]] #integration limit for kper(internal transverse momentum) and zf(fragmentation variable)

#####################______________________________###########################

#function to save results to csv file

def save_results(Pper, mean_L_1, sdev_L_1, mean_L_2, sdev_L_2, mean_L, sdev_L, total_time):

    # Define the output folder path
    output_folder = os.path.join("/users/swanismu/projects/sidislo/gamma_nucleus/x_g_scale/Posterior_uncer_xsection_EIC_A", f"output_EIC_Q2{Q2}_xbj{xbj}")
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # Create dynamic filename
    filename = os.path.join(output_folder, f"gammaA_L_Q2{Q2}_xbj{xbj}_zh{zh}__sam{sample_id}.csv")

    # Check if file exists
    file_exists = os.path.isfile(filename)

    # Open in append mode
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
    
        # If file is new, write header first
        if not file_exists:
            writer.writerow(["Pper", "mean_L_1", "sdev_L_1","mean_L_2", "sdev_L_2","mean_L", "sdev_L", "time_taken_sec"])

        writer.writerow([Pper,  mean_L_1, sdev_L_1, mean_L_2, sdev_L_2, mean_L, sdev_L, total_time ])

##################______________________________###########################

# --- results containers ---
mean_L_list_1 = []
sdev_L_list_1 = []

mean_L_list_2 = []
sdev_L_list_2= []

##################______________________________###########################
#Main SIDIS calculation loop over Pper values
for Q2 in Q2_vec:
    for xbj in xbj_vec:

        print(f"\n{'='*50}")
        print(f"Running A_L Q2={Q2}, xbj={xbj}, zh={zh}, sample={sample_id}")
        print(f"{'='*50}")

        for Pper in P_vec:           

        ##################______________________________###########################
            # Vegas integration setup 1

            intial_time_1 = time.time()

            #create vega integration object
            integ = vegas.Integrator(lims)
            
            #perform integration for longitudinal xsection
            integ(integrand_1, nitn=15, neval = 1e6, alpha = 0.5, nproc = 36) #warmup
            
            #initialize variables for mean and sdev
            mean_L_1, sdev_L_1 = 1e-15, 1e-15

            result_L_1 = integ(integrand_1, nitn=50, neval = 1e6, alpha = 0.5, nproc = 36)
            mean_L_1, sdev_L_1 = result_L_1.mean, result_L_1.sdev
            # store results
            mean_L_list_1.append(mean_L_1)
            sdev_L_list_1.append(sdev_L_1)

            final_time_1 = time.time()    
            
            print("--------------------------------------------------")
            print("Pper = ", Pper, "mean_L_1 = ", mean_L_1, "+/-", sdev_L_1, "Time taken:", final_time_1 - intial_time_1, "seconds")
                       
        ##################______________________________###########################
        
        ##################______________________________###########################
            # Vegas integration setup

            intial_time_2 = time.time()

            #create vega integration object
            integ = vegas.Integrator(lims_2)
            
            #perform integration for longitudinal xsection
            integ(integrand_2, nitn=15, neval = 1e6, alpha = 0.5, nproc = 36) #warmup
            
            #initialize variables for mean and sdev
            mean_L_2, sdev_L_2 = 1e-15, 1e-15

            result_L_2 = integ(integrand_2, nitn=50, neval = 1e6, alpha = 0.5, nproc = 36)
            mean_L_2, sdev_L_2 = result_L_2.mean, result_L_2.sdev
            # store results
            mean_L_list_2.append(mean_L_2)
            sdev_L_list_2.append(sdev_L_2)

            final_time_2 = time.time()

            print("Pper = ", Pper, "mean_L_2 = ", mean_L_2, "+/-", sdev_L_2, "Time taken:", final_time_2 - intial_time_2, "seconds")         

            ##########################################################################
            #combine both parts

            mean_L = mean_L_1 + mean_L_2
            sdev_L = np.sqrt(sdev_L_1**2 + sdev_L_2**2)
            total_time = (final_time_1 - intial_time_1) + (final_time_2 - intial_time_2)

            print("Pper = ", Pper, "mean_L = ", mean_L, "+/-", sdev_L, "Total Time taken:", total_time, "seconds")
            print("--------------------------------------------------")

            save_results(Pper, mean_L_1, sdev_L_1, mean_L_2, sdev_L_2, mean_L, sdev_L, total_time)
    
           