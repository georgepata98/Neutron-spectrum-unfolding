import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel
from deap import base, creator, tools, algorithms
import os

# ==== GRNN ====
class GRNN:
    def __init__(self, sigma=0.15):
        self.sigma = sigma

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        def gaussian(dist):
            return np.exp(- (dist ** 2) / (2 * self.sigma ** 2))

        preds = []
        for x in X:
            dists = np.linalg.norm(self.X_train - x, axis=1)
            weights = gaussian(dists)
            weighted_sum = np.dot(weights, self.y_train)
            preds.append(weighted_sum / np.sum(weights))
        return np.array(preds)

# ==== RBF ====
class RBFNetwork:
    def __init__(self, spread=0.15):
        self.spread = spread

    def fit(self, X, Y):
        self.centers = X
        G = rbf_kernel(X, self.centers, gamma=1.0 / (2 * self.spread ** 2))
        self.weights = np.linalg.pinv(G) @ Y

    def predict(self, X):
        G = rbf_kernel(X, self.centers, gamma=1.0 / (2 * self.spread ** 2))
        return G @ self.weights

# ==== GA ====
def run_ga(C_target, phi_true, R):
    def fitness(ind):
        phi_candidate = np.array(ind)
        C_calc = R @ phi_candidate
        mse = np.mean((C_target - C_calc) ** 2)
        smooth = np.sum((np.diff(phi_candidate, 2)) ** 2)
        return (1 / (1 + mse + 0.1 * smooth),)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.rand)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=phi_true.shape[0])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness)
    toolbox.register("mate", tools.cxBlend, alpha=0.2)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=80)
    for _ in range(50):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.7, mutpb=0.2)
        fits = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit
        pop = toolbox.select(offspring, k=len(pop))

    best = tools.selBest(pop, 1)[0]
    return np.array(best)

# ==== GUI ====
class UnfoldApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Neutron Spectrum Unfolding")
        self.geometry("900x700")

        # Method selection
        self.method_var = tk.StringVar(value="GRNN")
        ttk.Label(self, text="Choose Unfolding Method:").pack(pady=5)
        ttk.Combobox(self, textvariable=self.method_var, values=["GRNN", "RBF", "GA"]).pack()

        # Load buttons
        ttk.Button(self, text="Load Response Matrix (R)", command=self.load_response).pack(pady=2)
        ttk.Button(self, text="Load Spectra (phi)", command=self.load_spectra).pack(pady=2)

        # Run button
        ttk.Button(self, text="Run Unfolding", command=self.run_unfolding).pack(pady=10)

        # Plotting area
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

    def load_response(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV or NPY", "*.csv *.npy")])
        if not file_path:
            return
        try:
            if file_path.endswith('.csv'):
                self.R = np.loadtxt(file_path, delimiter=',')
            else:
                self.R = np.load(file_path)
            self.try_generate_counts()
            messagebox.showinfo("Success", f"Loaded response matrix: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_spectra(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV or NPY", "*.csv *.npy")])
        if not file_path:
            return
        try:
            if file_path.endswith('.csv'):
                self.phi = np.loadtxt(file_path, delimiter=',')
            else:
                self.phi = np.load(file_path)
            self.try_generate_counts()
            messagebox.showinfo("Success", f"Loaded spectra: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def try_generate_counts(self):
        if hasattr(self, 'R') and hasattr(self, 'phi'):
            self.C = self.phi @ self.R.T
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.C, self.phi, test_size=0.25, random_state=42
            )

    def run_unfolding(self):
        method = self.method_var.get()

        # If user hasn't loaded data, use synthetic
        if not hasattr(self, 'X_test'):
            np.random.seed(0)
            self.phi = np.abs(np.random.rand(251, 53))
            self.R = np.abs(np.random.rand(8, 53))
            self.C = self.phi @ self.R.T
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.C, self.phi, test_size=0.25, random_state=42
            )

        # Use first sample
        idx = 0
        C_input = self.X_test[idx]
        true_spectrum = self.y_test[idx]

        try:
            if method == "GRNN":
                model = GRNN()
                model.fit(self.X_train, self.y_train)
                pred = model.predict([C_input])[0]

            elif method == "RBF":
                model = RBFNetwork()
                model.fit(self.X_train, self.y_train)
                pred = model.predict([C_input])[0]

            elif method == "GA":
                pred = run_ga(C_input, true_spectrum, self.R)

            else:
                messagebox.showerror("Error", "Unknown method selected.")
                return
        except Exception as e:
            messagebox.showerror("Computation Error", str(e))
            return

        # Plot result
        self.ax.clear()
        self.ax.plot(true_spectrum, label="True Spectrum")
        self.ax.plot(pred, '--', label=f"Unfolded ({method})")
        self.ax.set_title("Neutron Spectrum Unfolding Result")
        self.ax.set_xlabel("Energy Group")
        self.ax.set_ylabel("Flux")
        self.ax.legend()
        self.canvas.draw()

if __name__ == "__main__":
    app = UnfoldApp()
    app.mainloop()
