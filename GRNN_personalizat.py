import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

random_gen = True  # generare de date random
if random_gen == True:
    def generate_random_data(n_samples, n_energy_groups):
        """ Se genereaza date in [0,1), argumentii din rand() sunt nr. de linii, respectiv coloane
        n_samples=nr. linii; n_energy_groups=nr. coloane """
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
    C = phi @ R  # daca pe linii este R(E), fiecare coloana fiind un alt detector

""" --- Functia predefinita train_test_split() ---
1) X_train si y_train sunt folosite pentru a antrena modelul GRNN
2) X_test si y_test sunt pentru a evalua performanta modelului
3) C = date de input (ratele de numarare calculate mai sus)
4) phi = date tinta (spectrele de neutroni care vor sa fie prezise)
5) test_size = 0.25 (25% date de test)
Observatii: variabilele X_train, X_test iau valori din datele de input C (25% vor fi in X_test); analog y_train si y_test din datele de output phi
"""
X_train, X_test, y_train, y_test = train_test_split(C, phi, test_size=0.25, random_state=42)

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
            return np.exp(- (dist ** 2) / (2 * self.sigma ** 2))
        
        print("len(X) =", len(X))
        print("len(X_train) =", len(X_train))
        
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
grnn = GRNN(sigma=0.15)
grnn.fit(X_train, y_train)
y_pred = grnn.predict(X_test)  # valorile prezise ale lui y pt. X_test
print("len(y_pred) =", len(y_pred))  # matrice cu 63 linii
print("len(y_test) =", len(y_test))  # matrice cu 63 linii

""" Evaluare abatere patratica medie (mse) """
mse = mean_squared_error(y_test, y_pred)
print(f"Mean squared error: {mse:.4e}")

""" Plotare spectru solutie """
index = 62  # sunt 63 in total, incepand de la 0
plt.plot(y_test[index], label='True Spectrum')
plt.plot(y_pred[index], '--', label='Unfolded Spectrum (GRNN)')
plt.xlabel("Energy Group")
plt.ylabel("Fluence [a.u.]")
plt.title("Neutron Spectrum Unfolding")
plt.legend()
plt.show()
