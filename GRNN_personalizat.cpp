#include <iostream>
#include <random>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;


double gaussian(double dist, double sigma) {
    return exp(-(pow(dist,2)) / (2*pow(sigma,2)));
}


int main()
{
bool random_gen = false;  // generare de date random

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
    double X_train[n_samples_train][n_detectors], X_test[n_samples_test][n_detectors], y_train[n_samples_train][n_energy_groups], y_test[n_samples_test][n_energy_groups];  // !!! y_test[i] sunt folositi la determinarea MSE pt. imbunatatirea lui sigma

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
    /*
    ATENTIE:  matricea de raspuns din .csv are pe linii raspunsul functie de grupul de energie si pe coloane tipul de detector (invers de cum ar trebui),
              si la fel si fluentele din .csv.
           :  valorile care reprezinta spectrele de neutroni phi(E) in IAEA TRS403 sunt fluentele de grup impartite la intervalul de letargie
              (i.e. ln(E_i+1) - ln(E_i)).
           :  in fisierele .csv prima coloana este energia in [eV].
    */

    // --------------------------------------
    // | Citirea datelor din fisierele .csv |
    // --------------------------------------
    ifstream ifile;
    string filename_phi, filename_R;
    int n_samples = 0, n_energy_groups = 0, n_detectors = 0;

    cout << "Nume fisier .csv cu fluentele de neutroni (fara extensia .csv): "; cin >> filename_phi;
    cout << "Nume fisier .csv cu matricea de raspuns (fara extensia .csv): "; cin >> filename_R;

    // 1. Extragerea datelor pentru phi
    vector<vector<double>> phi_csv;  // matricea phi din fisierul .csv
    ifile.open((filename_phi + ".csv").c_str());
    if(!ifile) { cout << filename_phi << ".csv nu exista." << endl; }
    if(ifile)
    {
        int aux = 0;
        string line;

        while(getline(ifile, line))
        {
            vector<double> row;
            stringstream ss(line);
            string value;
            while(getline(ss, value, ','))
            {
                row.push_back(stod(value));  // stod converteste string in double;  stoi converteste string in int
                if(aux == 0) { n_samples++; }
            }
            phi_csv.push_back(row);  // adauga intreaga linie in matricea phi_csv
            if(aux == 0) { aux = 1; }
            n_energy_groups++;
        }

        ifile.close();
    }
    // 2. Extragerea datelor pentru R
    vector<vector<double>> R_csv;  // matricea R din fisierul .csv
    ifile.open((filename_R + ".csv").c_str());
    if(!ifile) { cout << filename_R << ".csv nu exista." << endl; }
    if(ifile)
    {
        int aux = 0;
        string line;

        while(getline(ifile, line))
        {
            vector<double> row;
            stringstream ss(line);
            string value;
            while(getline(ss, value, ','))
            {
                row.push_back(stod(value));
                if(aux == 0) { n_detectors++; }
            }
            R_csv.push_back(row);
            if(aux == 0) { aux = 1; }
        }

        ifile.close();
    }

    // -----------------------------------------------------------------
    // | Convertirea lui phi din unit. de letargie in unit. de energie |
    // -----------------------------------------------------------------
    double phi_converted[n_energy_groups-1][n_samples-1], R_converted[n_energy_groups-1][n_detectors-1];
    // 1. Convertire phi in unit. de energie si scaderea dimensiunii
    for(int i=0; i<n_energy_groups; i++)
    {
        for(int j=0; j<n_samples; j++)
        {
            if(j>0 && i<n_energy_groups-1)  // skip coloana cu energii si linia cu upper bin
            {
                phi_converted[i][j-1] = phi_csv[i][j] * (log(phi_csv[i+1][0]) - log(phi_csv[i][0]));
            }
        }
    }
    // 2. Scaderea dimensiunii lui R cu 1 linie si 1 coloana
    for(int i=0; i<n_energy_groups; i++)
    {
        for(int j=0; j<n_detectors; j++)
        {
            if(j>0 && i<n_energy_groups-1)  // skip coloana cu energii si linia cu upper bin
            {
                R_converted[i][j-1] = R_csv[i][j];
            }
        }
    }

    // ---------------------------
    // | Incarcarea lui phi si R |
    // ---------------------------
    n_samples = n_samples - 1;              // prima coloana din .csv fiind energia
    n_energy_groups = n_energy_groups - 1;  // ultima linie din .csv fiind upper bin-ul
    n_detectors = n_detectors - 1;
    double phi[n_samples][n_energy_groups], R[n_detectors][n_energy_groups];
    // phi_converted si R_converted au aceeasi dimensiune ca phi si R, dar sunt transpusele lor
    // 1. Incarcarea lui phi
    for(int i=0; i<n_energy_groups; i++)
    {
        for(int j=0; j<n_samples; j++)
        {
            phi[j][i] = phi_converted[i][j];
        }
    }
    // 2. Incarcarea lui R
    for(int i=0; i<n_energy_groups; i++)
    {
        for(int j=0; j<n_detectors; j++)
        {
            R[j][i] = R_converted[i][j];
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
    double X_train[n_samples_train][n_detectors], X_test[n_samples_test][n_detectors], y_train[n_samples_train][n_energy_groups], y_test[n_samples_test][n_energy_groups];  // !!! y_test[i] sunt folositi la determinarea MSE pt. imbunatatirea lui sigma

    // ---------------------------------------
    // | Incarcarea datelor de train si test |
    // ---------------------------------------
    random_device rd;
    mt19937 gen(rd());  // Mersenne Twister RNG
    uniform_real_distribution<> dis(0.0, 1.0);  // interval [0,1)
    double random_value = dis(gen);
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


/*
Observatii:  C[i][j] = nr. de counturi produse cu spectrul i de detectorul j.
          :  nr. neuroni din pattern layer = nr. training samples.
*/
return 0;
}
