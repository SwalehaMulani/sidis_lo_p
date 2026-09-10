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
parser.add_argument("--sample", type=int, required=True, help="Posterior sample index")


args = parser.parse_args()

#parameters
zh = 0.5 #Longitudinal momentum fraction of hadron(k1+/P+)
Q2_vec = [5, 10, 50] #GeV^2
xbj_vec = [0.01, 0.008, 0.006, 0.004, 0.002, 0.001, 0.0008, 0.0006, 0.0004, 0.0002, 0.0001] #Bjorken x values to be taken from EIC data kinematic bins
Nc = 3
aem = 1/137
mq = 0.14 #GeV light quark mass

data = pd.read_csv("/users/swanismu/projects/sidislo/gamma_proton/x_g_scale/Qs2_red_curve_10pts.csv")

P_vec = np.linspace(1,15,57) #GeV

############____________________________###############################
#option1 : extract sigma0_2 from the posterior parameter file for the sample index
posterior_file = "/users/swanismu/projects/sidislo/bk_momentum_posterior_10-3_P/posteriorsamples_100_LOmvefit.dat"
posterior_data = np.loadtxt(posterior_file, comments="#") #load the posterior samples from the file and skip any comment lines starting with '#'
sample_id = args.sample
sigma0_2 = posterior_data[sample_id, 3]
#sigma0 = 2.0 * sigma0_2 * 2.56819 #convert from mb to GeV^-2
sigma0 = 2.0 * sigma0_2 # mb
print("Using sigma0_2 =", sigma0_2, "mb from posterior sample index", sample_id)
#________________________________________________________________________
#option2 : extract sigma0_2 directly from the corresponding BK file for the sample index
# def extract_sigma0_2(filename):
#     with open(filename, "r") as f:
#         for line in f:
#             if "sigma0/2" in line:
#                 # Extract scientific number
#                 match = re.search(r"sigma0/2\s*=\s*([0-9.eE+-]+)", line)
#                 if match:
#                     return float(match.group(1))
#     raise ValueError("sigma0/2 not found in file")
#sample_id = args.sample




#define th functions required for the integrand of xsection
bk_dir = "/users/swanismu/projects/sidislo/bk_momentum_posterior_10-3_P/10e-3min"

bk_file = os.path.join(bk_dir, f"2Dft_{sample_id}.csv")

#sigma0_2 = extract_sigma0_2(bk_file)

#sigma0 = 2.0 * sigma0_2 * 2.56819  # mb → GeV^-2
#sigma0 = 2.0 * sigma0_2 # mb

print("Using sigma0_2 =", sigma0_2, "mb from file", bk_file)
#print("Converted sigma0 =", sigma0, "GeV^-2")

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

##########____________________________###############################

#fixing fragmentation function D(zf)

ff = lhapdf.getPDFSet("NNFF10_PIsum_lo").mkPDF(0)

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
    output_folder = os.path.join("/users/swanismu/projects/sidislo/gamma_proton/x_g_scale/Posterior_uncer_xsection_EIC_P", f"output_EIC_Q2{Q2}_xbj{xbj}")


    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create dynamic filename
    filename = os.path.join(output_folder, f"gammaP_L_Q2{Q2}_xbj{xbj}_zh{zh}__sam{sample_id}.csv")

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

#define xsection integrand
@vegas.rbatchintegrand
def SIDIS_integrand(kper, zf):
    z1 = zh / zf #z1 is longitudinal momentum fraction of quark z1= k1+/(k1+ + k2+) fixed by zh and zf
    valid = (z1 > 0) & (z1 < 1)
    z2 = 1 - z1 #z2 is fixed by z1 only for this code for simplicity
    arg = Q2 * z1 * z2 + mq**2
    eps = np.sqrt(np.where(valid, arg, 0.0))
    H =(aem * Nc * sigma0 ) / (2*(2*np.pi)**4)
    Pquark = Pper/zf
    M = kper *(1/(eps**2 + Pquark**2)**2 - ((eps**2 + Pquark**2 + kper**2)*(eps**2 + Pquark**2 + 2 * kper**2) - 8 * kper**2 * Pquark**2)/((eps**2 + Pquark**2)*((eps**2 + Pquark**2 + kper**2)**2 - 4 * kper**2 * Pquark**2)**(3/2)))
    Z = z1*(1-z1)* eps**2 * 8 #z1 here is longititudinal momentum fraction of quark z1= k1+/(k1+ + k2+)
    D_light_sum = 2 * D_light(Pper, zf) #multiply by 2 for antiquark contribution for each flavor of that of quark
    dipole = dip(Pquark, kper, z1)
    result = H * Z * M * dipole*D_light_sum * (1/zf**3)  # multiplying with Pper to get d sigma / d Pper format similar to HERA data
    result = np.where(valid, result, 0.0)
    return  result # multiplying with Pper to get d sigma / d Pper format similar to HERA data

@vegas.rbatchintegrand
def integrand(x):
    kper, zf = x
    return SIDIS_integrand(kper, zf)
            
#integration limits
                
lims = [[0.01, 100], [0.01,1]] #integration limits for kper and zf

##########____________________________###############################
##########___________________________######################################
#Main SIDIS calculation loop over Pper values
for Q2 in Q2_vec:
    for xbj in xbj_vec:

        print(f"\n{'='*50}")
        print(f"Running Q2={Q2}, xbj={xbj}, zh={zh}, sample={sample_id}")
        print(f"{'='*50}")

        for Pper in P_vec:

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
            