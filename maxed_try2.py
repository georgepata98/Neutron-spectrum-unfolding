import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, dual_annealing


# --- Formulae from Giacomelli (2019) ---

def entropy(f, f_def):
    return -np.sum(f * np.log(f / f_def) + f_def - f)

def spectrum(R, f_def, lambdas):
    exponent = -np.dot(lambdas, R)
    exponent = np.clip(exponent, -700, 700)
    f = f_def * np.exp(exponent)
    return np.clip(f, 1e-12, 1e12)  # prevent underflow/overflow

def chi_squared(f, N, R, sigma):
    model = np.dot(R, f)
    model = np.clip(model, 1e-12, 1e12)
    return np.sum(((N - model) / sigma) ** 2)

def normalization_constraint(lambdas, R, f_def):
    f = spectrum(R, f_def, lambdas)
    return np.sum(f) - 1

def chi_squared_constraint(lambdas, N, R, sigma, f_def):
    f = spectrum(R, f_def, lambdas)
    return chi_squared(f, N, R, sigma) - 1

def objective(lambdas, N, R, sigma, f_def):
    f = spectrum(R, f_def, lambdas)
    S = entropy(f, f_def)
    chi2_penalty = 1e6 * abs(chi_squared(f, N, R, sigma) - 1)
    norm_penalty = 1e6 * abs(np.sum(f) - 1)
    return -S + chi2_penalty + norm_penalty


# --- Input data ---

nbins = 8
f_true = np.array([10, 30, 50, 80, 60, 40, 20, 10])  # true spectrum
f_def = np.ones(nbins)
R = np.array([
    [0.6, 0.9, 1.0, 0.5, 0.3, 0.1, 0.05, 0.02],
    [0.2, 0.4, 0.9, 1.0, 0.8, 0.5, 0.2, 0.1],
    [0.05, 0.1, 0.3, 0.8, 1.0, 0.9, 0.6, 0.3],
    [0.01, 0.03, 0.1, 0.3, 0.8, 1.0, 0.9, 0.6]
])
N = R @ f_true
sigma = 0.05 * N
lambdas_initial = np.zeros(len(N))

# --- Constrained optimization using SLSQP ---

constraints = [
    {'type': 'eq', 'fun': chi_squared_constraint, 'args': (N, R, sigma, f_def)},
    {'type': 'eq', 'fun': normalization_constraint, 'args': (R, f_def)}
]

result_slsqp = minimize(
    fun=lambda lambdas: -entropy(spectrum(R, f_def, lambdas), f_def),
    x0=lambdas_initial,
    method='SLSQP',
    constraints=constraints
)

optimal_lambdas_slsqp = result_slsqp.x
f_unfolded_slsqp = spectrum(R, f_def, optimal_lambdas_slsqp)
predicted_N_slsqp = R @ f_unfolded_slsqp
entropy_slsqp = entropy(f_unfolded_slsqp, f_def)

# --- Simulated Annealing ---

bounds = [(-5, 5)] * len(N)
result_sa = dual_annealing(
    func=objective,
    bounds=bounds,
    args=(N, R, sigma, f_def)
)

optimal_lambdas_sa = result_sa.x
f_unfolded_sa = spectrum(R, f_def, optimal_lambdas_sa)
predicted_N_sa = R @ f_unfolded_sa
entropy_sa = entropy(f_unfolded_sa, f_def)

# --- Results ---

print("\n--- Results using SLSQP ---")
print("Optimal lambdas:", optimal_lambdas_slsqp)
print("Unfolded spectrum f_k:", f_unfolded_slsqp)
print("Sum f_k:", np.sum(f_unfolded_slsqp))
print("Chi^2:", chi_squared(f_unfolded_slsqp, N, R, sigma))

print("\n--- Results using Simulated Annealing ---")
print("Optimal lambdas:", optimal_lambdas_sa)
print("Unfolded spectrum f_k:", f_unfolded_sa)
print("Sum f_k:", np.sum(f_unfolded_sa))
print("Chi^2:", chi_squared(f_unfolded_sa, N, R, sigma))


# --- Plot setup ---

plot = False
if plot == True:
    # Normalize all spectra
    f_true_norm = f_true / np.sum(f_true)
    f_slsqp_norm = f_unfolded_slsqp / np.sum(f_unfolded_slsqp)
    f_sa_norm = f_unfolded_sa / np.sum(f_unfolded_sa)

    bins = np.arange(1, nbins + 1)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # --- True Spectrum ---
    axs[0].plot(bins, f_true_norm, 'ko--', linewidth=2, markersize=6)
    axs[0].set_title('True Spectrum')
    axs[0].set_ylabel('Normalized Intensity')
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # --- SLSQP Result ---
    axs[1].plot(bins, f_slsqp_norm, 'b^-', linewidth=2, markersize=8)
    axs[1].set_title('Unfolded Spectrum (SLSQP)')
    axs[1].set_ylabel('Normalized Intensity')
    axs[1].grid(True, linestyle='--', alpha=0.5)

    # --- Simulated Annealing Result ---
    axs[2].plot(bins, f_sa_norm, 'rs-', linewidth=2, markersize=6)
    axs[2].set_title('Unfolded Spectrum (Simulated Annealing)')
    axs[2].set_xlabel('Energy Bin')
    axs[2].set_ylabel('Normalized Intensity')
    axs[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


""" --- Explicare program ---
1. Maximizarea entropiei:
   Functia entropy(f, f_def) calculeaza entropia spectrului f relativ la spectrul default f_def. Scopul metodei
   MaxEnt este de a maximiza aceasta entropie in timp ce satisface constrangerile.
2. Functia spectrum(R, f_def, lambdas) calculeaza spectrul f folosind matricea de raspuns R, spectrul default
   f_def si multiplicatorii Lagrange lambdas. Acesta este miezul metodei MaxEnt. Functia aceasta are ca scop 
   in loc sa gasim direct spectrul f, definim functia spectrum(R, f_def, lambdas) care transforma problema 
   intr-una a gasirii celor mai buni lambdas, astfel incat spectrul rezultat sa fiteze datele (i.e. R*f ~ N) 
   si sa aiba maximul de entropie.
3. Constrangerile: 3.1) de normalizare: se asigura ca spectrul prezis este insumat la 1.
                   3.2) de chi^2: valoarea lui chi^2 da aprox. 1.
4. Metoda de optimizare SA este o metoda globala de optimizare si tine cont de constrangeri prin intermediul 
   termenilor de penalty in functia objective() (se adauga penalty-uri mari daca chi^2!=1 sau sum(f)!=1)
"""