#include <iostream>
#include <random>
#include <cmath>
#include <fstream>

using namespace std;


double gaussian(double dist, double sigma) {
    return exp(-(pow(dist,2)) / (2*pow(sigma,2)));
}


int main()
{
bool random_gen = true;  // generare de date random

if(random_gen == true)
{
    random_device rd;
    mt19937 gen(rd());  // Mersenne Twister RNG
    uniform_real_distribution<> dis(0.0, 1.0);  // interval [0,1)
    double random_value = dis(gen);
        
    int n_samples = 251, n_energy_groups = 53, n_detectors = 8;
    double phi[n_samples][n_energy_groups], R[n_detectors][n_energy_groups];

    // ---------------------
    // | Incarcare spectre |
    // ---------------------
    for(int i=0; i<n_samples; i++)
    {
        for(int j=0; j<n_energy_groups; j++)
        {
            phi[i][j] = dis(gen);
        }
    }
    // ---------------------------------
    // | Incarcare matricea de raspuns |
    // ---------------------------------
    for(int i=0; i<n_detectors; i++)
    {
        for(int j=0; j<n_energy_groups; j++)
        {
            R[i][j] = dis(gen);
        }
    }

    // -------------------------------
    // | Determinare nr. de counturi |
    // -------------------------------
    // C = phi @ R.T (produs matriceal dintre phi si R transpus)
    double C[n_samples][n_detectors];
    for(int i=0; i<n_samples; i++)
    {
        for(int j=0; j<n_detectors; j++)
        {
            double sum = 0;
            for(int k=0; k<n_energy_groups; k++)
            {
                sum += phi[i][k] * R[j][k];
            }
            C[i][j] = sum;
        }
    }

    // ----------------------------------------
    // | Dimensiunea datelor de train si test |
    // ----------------------------------------
    //   C[i][j] = datele de input
    // phi[i][j] = datele tinta
    double test_size = 0.25;  // % test data
    bool user_test_size = false;
    if(user_test_size == true)
    {
        cout << "test_size (>0 si <1) = "; cin >> test_size;
    }
    while(test_size > 1 || test_size < 0)
    {
        cout << "Error: test_size is not a percentage!" << endl;
        cout << "test_size = "; cin >> test_size;
    }
    int n_samples_train = (1 - test_size) * n_samples;
    int n_samples_test = n_samples - n_samples_train;
    double X_train[n_samples_train][n_detectors], X_test[n_samples_test][n_detectors], y_train[n_samples_train][n_energy_groups], y_test[n_samples_test][n_energy_groups];  // !!! y_test[i] sunt folositi la determinarea MSE pt. imbunatatirea lui sigma[i]

    // ---------------------------------------
    // | Incarcarea datelor de train si test |
    // ---------------------------------------
    int i_train = 0, i_test = 0;
    for(int i=0; i<n_samples; i++)
    {
        random_value = dis(gen);
        if(random_value < (1 - test_size) && i_train < n_samples_train)
        {
            for(int j=0; j<n_detectors; j++)
            {
                X_train[i_train][j] = C[i][j];
            }
            for(int j=0; j<n_energy_groups; j++)
            {
                y_train[i_train][j] = phi[i][j];
            }
            i_train++;
        }
        else if(i_test < n_samples_test)
        {
            for(int j=0; j<n_detectors; j++)
            {
                X_test[i_test][j] = C[i][j];
            }
            for(int j=0; j<n_energy_groups; j++)
            {
                y_test[i_test][j] = phi[i][j];
            }
            i_test++;
        }
    }

    // -----------------------------------------------
    // | Algoritmul Generalized Regression NN (GRNN) |
    // -----------------------------------------------
    // 1. Calcularea distantelor si ponderilor (w) din pattern layer
    double dists[n_samples_test][n_samples_train], weights[n_samples_test][n_samples_train];  // nr. neuroni patter layer = nr. training samples
    double y_pred[n_samples_test][n_energy_groups];  // valorile prezise pentru fiecare X_test[i]
    double sigma;
    // --- Sigma variabil ---
    bool mse_plot = false;
    if(mse_plot == true)  // plotare MSE(sigma) pentru a vedea sigma care minimizeaza MSE
    {
        ofstream msefile;  // mse functie de sigma
        msefile.open("mse_data.txt");
        for(sigma=0.1; sigma<5; sigma+=0.01)  // Wang et al., 2019
        {
            for(int i=0; i<n_samples_test; i++)
            {
                for(int j=0; j<n_samples_train; j++)
                {
                    double sum = 0;
                    for(int k=0; k<n_detectors; k++)
                    {
                        sum += (X_test[i][k] - X_train[j][k]) * (X_test[i][k] - X_train[j][k]);
                    }
                    dists[i][j] = sqrt(sum);
                    weights[i][j] = gaussian(dists[i][j], sigma);
                }
            }
            // 2. Calcularea vectorilor de output pentru fiecare X_test[i]
            for(int i=0; i<n_samples_test; i++)
            {
                for(int j=0; j<n_energy_groups; j++)
                {
                    double numarator = 0;
                    double numitor = 0;
                    for(int k=0; k<n_samples_train; k++)
                    {
                        numarator += weights[i][k] * y_train[k][j];
                        numitor += weights[i][k];
                    }
                    y_pred[i][j] = numarator / numitor;
                }
            }
            // 3. Determinarea MSE dintre matricile y_test si y_pred
            double mse = 0;
            for(int i=0; i<n_samples_test; i++)
            {
                for(int j=0; j<n_energy_groups; j++)
                {
                    mse += (y_test[i][j] - y_pred[i][j]) * (y_test[i][j] - y_pred[i][j]);
                }
            }
            mse = mse / (n_samples_test * n_energy_groups);
            msefile << mse << " " << sigma << endl;
        }
        msefile.close();
    }
    else
    {
        sigma = 0.15;
        for(int i=0; i<n_samples_test; i++)
        {
            for(int j=0; j<n_samples_train; j++)
            {
                double sum = 0;
                for(int k=0; k<n_detectors; k++)
                {
                    sum += (X_test[i][k] - X_train[j][k]) * (X_test[i][k] - X_train[j][k]);
                }
                dists[i][j] = sqrt(sum);
                weights[i][j] = gaussian(dists[i][j], sigma);
            }
        }
        // 2. Calcularea vectorilor de output pentru fiecare X_test[i]
        for(int i=0; i<n_samples_test; i++)
        {
            for(int j=0; j<n_energy_groups; j++)
            {
                double numarator = 0;
                double numitor = 0;
                for(int k=0; k<n_samples_train; k++)
                {
                    numarator += weights[i][k] * y_train[k][j];
                    numitor += weights[i][k];
                }
                y_pred[i][j] = numarator / numitor;
            }
        }
        // 3. Determinarea MSE dintre matricile y_test si y_pred
        double mse = 0;
        for(int i=0; i<n_samples_test; i++)
        {
            for(int j=0; j<n_energy_groups; j++)
            {
                mse += (y_test[i][j] - y_pred[i][j]) * (y_test[i][j] - y_pred[i][j]);
            }
        }
        mse = mse / (n_samples_test * n_energy_groups);
    }
    

    // ---------------------
    // | Salvare rezultate |
    // ---------------------
    bool save = true;
    if(save == true)
    {
        ofstream ofile;
        int index;

        cout << "Enter the index of X_test[i] vector that you want to predict: "; cin >> index;
        while(index<0 || index>=n_samples_test)
        {
            cout << "X_test[" << index << "] doesn't exist.\nindex = "; cin >> index;
        }
        // --- Incarcarea y_test[index], y_train[index] intr-un fisier de output ---
        if(index>=0 && index<n_samples_test)
        {
            string filename = ("data_sample" + std::to_string(index) + ".txt").c_str();

            ofile.open(filename);
            for(int j=0; j<n_energy_groups; j++)
            {
                if(j == 0) { ofile << "y_test  ,  y_pred" << endl; }
                ofile << y_test[index][j] << " " << y_pred[index][j] << endl;
            }
            ofile.close();
            cout << "File " + filename + " created successfully." << endl;
        }
    }
}

// =========================================
// || SPECTRE DE NEUTRONI DIN IAEA TRS403 ||
// =========================================
else  
{

}


/*
Observatii:  C[i][j] = nr. de counturi produse cu spectrul i de detectorul j
          :  nr. neuroni din pattern layer = nr. training samples
*/
return 0;
}