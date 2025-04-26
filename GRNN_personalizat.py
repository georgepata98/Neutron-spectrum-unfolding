import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt


random_gen = False
if random_gen == True:
    """ Generarea de date random pentru phi si R """
    def generate_random_data(n_samples, n_energy_groups):
        """ Se genereaza date in [0,1), argumentii din rand() sunt nr. de linii, respectiv coloane
        n_samples = nr. linii; n_energy_groups = nr. coloane """
        spectrum = np.abs(np.random.rand(n_samples, n_energy_groups))  # spectrul simulat
        return spectrum

    def response_matrix(n_detectors, n_energy_groups):
        """ Matricea de raspuns cu n_detectors detectori """
        response = np.abs(np.random.rand(n_detectors, n_energy_groups))
        return response

    np.random.seed(0)  # se produce aceeasi secventa de numere aleatoare de fiecare data cand codul este rulat
    n_samples = 251       # numarul de spectre
    n_energy_groups = 53  # numarul de bini
    n_detectors = 8       # numarul de detectori
    phi = generate_random_data(n_samples, n_energy_groups)
    R = response_matrix(n_detectors, n_energy_groups)
    C = phi @ R.T  # nr. de counturi (R.T = R transpus)

else:
    """ Spectre de neutroni din IAEA TRS403 """
    filename_phi = input("Nume fisier .csv cu fluentele de neutroni (fara extensia .csv): ")
    filename_R = input("Nume fisier .csv cu matricea de raspuns (fara extensia .csv): ")
    phi_csv, R_csv, n_samples, n_energy_groups, n_detectors = [], [], 0, 0, 0

    ifile = open(filename_phi + ".csv", "r")
    for i, row in enumerate(ifile):  # i = nr. linie
        values = row.strip().split(",")
        float_values = [float(col) for col in values if col.strip()]
        if i == 0:
            n_samples = len(float_values)
        phi_csv.append(float_values)
        n_energy_groups += 1
    ifile.close()

    ifile = open(filename_R + ".csv", "r")
    for i, row in enumerate(ifile):  # i = nr. linie
        values = row.strip().split(",")
        float_values = [float(col) for col in values if col.strip()]
        if i == 0:
            n_detectors = len(float_values)
        R_csv.append(float_values)
    ifile.close()

    phi_csv = np.array(phi_csv)  # transformarea din lista in np.array()
    R_csv = np.array(R_csv)      # transformarea din lista in np.array()
    phi_converted = np.zeros((n_energy_groups-1, n_samples-1))  # umplere initiala cu 0
    R_converted = np.zeros((n_energy_groups-1, n_detectors-1))  # umplere initiala cu 0
    """ Convertire phi din unit. de letargie in unit. de energie si scadere dimensiuni """
    for i in range(n_energy_groups):
        for j in range(n_samples):
            if j>0 and i<n_energy_groups-1:  # skip coloanei cu energii si a liniei cu upper bin
                phi_converted[i][j-1] = phi_csv[i][j] * (np.log(phi_csv[i+1][0]) - np.log(phi_csv[i][0]))
        for j in range(n_detectors):
            if j>0 and i<n_energy_groups-1:  # skip coloanei cu energii si a liniei cu upper bin
                R_converted[i][j-1] = R_csv[i][j]

    n_samples -= 1        # decrementarea lui n_samples cu 1
    n_energy_groups -= 1  # decrementarea lui n_energy_groups cu 1
    n_detectors -= 1      # decrementarea lui n_detectors cu 1
    phi = np.zeros((n_samples, n_energy_groups))
    R = np.zeros((n_detectors, n_energy_groups))
    """ Incarcarea lui phi si R """
    for i in range(n_energy_groups):
        for j in range(n_samples):
            phi[j][i] = phi_converted[i][j]
        for j in range(n_detectors):
            R[j][i] = R_converted[i][j]

    """ Calcularea numarului de counturi C """
    C = phi @ R.T
    
    """ Introducerea manuala a counturilor experimentale C_exp """
    experimental_data = False
    if experimental_data == True:
        C_exp = []
        print(f"Introduceti valorile afisate de cei {n_detectors} detectori:")
        for i in range(n_detectors):
            val = float(input(f"C[{i}] = "))  # converteste input in float
            C_exp.append(val)
        C_exp = np.array(C_exp)  # transformare din lista [] in np.array()


X_train, X_test, y_train, y_test = train_test_split(C, phi, test_size=0.25, random_state=42)
# C = date de input (X_train + X_test)
# phi = date de output (y_train + y_test)


""" Clasa Generalized Regression Neural Network (GRNN) """
class GRNN:
    def __init__(self, sigma=0.1):
        self.sigma = sigma
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        def gaussian(dist):
            return np.exp(-(dist ** 2) / (2 * self.sigma ** 2))
        
        preds = []
        for x in X:
            """ --- Algoritmul GRNN --- """
            dists = np.linalg.norm(self.X_train - x, axis=1)
            weights = gaussian(dists)
            weighted_sum = np.dot(weights, self.y_train)
            norm = np.sum(weights)
            preds.append(weighted_sum / norm)
        return np.array(preds)

""" Antrenare si testare GRNN """
grnn = GRNN(sigma=0.15)  # cea mai optima valoare a lui sigma
grnn.fit(X_train, y_train)
y_pred = grnn.predict(X_test)  # valorile prezise ale lui X_test
if experimental_data == True:
    y_exp = grnn.predict(C_exp)

""" Evaluare abatere patratica medie (mse) """
mse = mean_squared_error(y_test, y_pred)

""" Graficul Abaterea patratica medie vs. sigma """
mse_plot = False
if mse_plot == True:
    mse_array = []    # mse pentru fiecare sigma
    sigma_array = []  # array cu toti sigma

    for sigma in np.arange(0.1, 3, 0.01):
        grnn = GRNN(sigma=sigma)
        grnn.fit(X_train, y_train)
        y_pred = grnn.predict(X_test)
        mse_array.append(mean_squared_error(y_test, y_pred))
        sigma_array.append(sigma)

if mse_plot == False:  # printare mse dintre y_test si y_pred numai cand valoarea lui mse este unica si testeaza capacitatea modelului de a prezice (testarea asemanarii dintre y_test (i.e. valorile lui y cunoscute) si y_pred (i.e. valorile lui y prezise))
    print(f"Mean squared error: {mse:.4e}")

""" Plotare spectru solutie """
if experimental_data == False and mse_plot == False:
    index = int(input("Indexul lui X_test[i] al carui y_test[i] doriti sa il plotati: "))
    plt.plot(y_test[index], label='True Spectrum', color='red')
    plt.plot(y_pred[index], '--', label='Unfolded Spectrum (GRNN)', color='blue')
    plt.xlabel("Energy Group")
    plt.ylabel("Fluence [a.u.]")
    plt.title("True vs. Unfolded Spectrum")
    plt.legend()
    plt.show()
if experimental_data == True and mse_plot == False:
    plt.plot(y_exp[0], label='Unfolded Spectrum (GRNN)', color='blue')
    plt.xlabel("Energy Group")
    plt.ylabel("Fluence [a.u.]")
    plt.title("Neutron Spectrum")
    plt.legend()
    plt.show()
if mse_plot == True:
    plt.plot(sigma_array, mse_array, label='mse(sigma)', color='green')
    plt.xlabel("Smoothing parameter, sigma")
    plt.ylabel("Mean Squared Error")
    plt.title("Mean Squared Error vs. Smoothing Parameter")
    plt.legend()
    plt.show()