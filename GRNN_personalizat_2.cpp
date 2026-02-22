/*
READ-ME !:
- Programul se foloseste pentru spectroscopie experimentala, cu rate de counturi corectate de efectele de retroimprastiere.
- Matricea R_csv initiala trebuie sa aiba acelasi nr. de coloane cu nr. detectorilor folositi la experiment (neglijandu-se coloana 1 cu energiile), si diametrele detectorilor sa fie aceleasi.
- Coloanele de date de antrenare trebuie sa fie cu aceleasi tipuri de detectori (diametre si model) ca cele din experiment.
*/
#include <iostream>
#include <random>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

double gaussian(double dist, double sigma)
{
    return exp(-(pow(dist,2)) / (2*pow(sigma,2)));
}

int main()
{
    ifstream ifile;
    string filename_phi, filename_R;
    int n_samples = 0, n_energy_groups = 0, n_detectors = 0;

    cout << "Nume fisier .csv cu FLUENTELE de neutroni (fara extensia .csv): "; cin >> filename_phi;
    cout << "Nume fisier .csv cu matricea de Raspuns (fara extensia .csv): "; cin >> filename_R;

    // Extragerea matricilor phi si R din fisierele csv

    vector<vector<double>> phi_csv;
    vector<vector<double>> R_csv;
    
    ifile.open((filename_phi + ".csv").c_str());
    if(!ifile) { cout << "Nu s-a putut deschide fisierul " << filename_phi << ".csv" << endl; }
    else
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
                if(aux == 0) { n_samples++; }
            }
            phi_csv.push_back(row);
            if(aux == 0) { aux = 1; }
            n_energy_groups++;
        }
        ifile.close();
    }

    ifile.open((filename_R + ".csv").c_str());
    if(!ifile) { cout << "Nu s-a putut deschide fisierul " << filename_R << ".csv" << endl; }
    else
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

	/* - Convertire phi din unitati de letargie in unitati de energie si salvarea noului phi, cu dimensiune redusa;
       - Salvarea noului R, cu dimensiune redusa.
	*/
    double phi_converted[n_energy_groups-1][n_samples-1], R_converted[n_energy_groups-1][n_detectors-1];

    for(int i=0; i<n_energy_groups; i++)
    {
        for(int j=0; j<n_samples; j++)
        {
            if(j>0 && i<n_energy_groups-1)
            {
                phi_converted[i][j-1] = phi_csv[i][j] * (log(phi_csv[i+1][0]) - log(phi_csv[i][0]));
            }
        }
    }
    for(int i=0; i<n_energy_groups; i++)
    {
        for(int j=0; j<n_detectors; j++)
        {
            if(j>0 && i<n_energy_groups-1)
            {
                R_converted[i][j-1] = R_csv[i][j];
            }
        }
    }

    n_samples -= 1;
    n_energy_groups -= 1;
    n_detectors -= 1;
    double phi[n_samples][n_energy_groups], R[n_detectors][n_energy_groups];

    for(int i=0; i<n_energy_groups; i++)
    {
        for(int j=0; j<n_samples; j++)
        {
            phi[j][i] = phi_converted[i][j];
        }
    }
    for(int i=0; i<n_energy_groups; i++)
    {
        for(int j=0; j<n_detectors; j++)
        {
            R[j][i] = R_converted[i][j];
        }
    }

    // Matricea de counturi C = phi * R.T

    double C[n_samples][n_detectors];
    ofstream ofile;

    ofile.open("C_matrix.csv");
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
            if(j < n_detectors - 1)
            {
                ofile << C[i][j] << ",";
            }
            if(j == n_detectors - 1)
            {
                ofile << C[i][j];
            }
        }
        ofile << "\n";
    }
    ofile.close();

    // GRNN

    double sigma;
    double X_exp[n_detectors], y_exp[n_energy_groups];  // ratele exp. corectate pt. fiecare detector si spectrul prezis y_exp
    double d[n_samples], w[n_samples];

    cout << "Valoarea parametrului sigma din functia gaussiana (de obicei = 0.15):\nsigma = "; cin >> sigma;

    for(int i=0; i<n_detectors; i++)
    {
        if(i == 0) { cout << "\nIntroduceti ratele experimentale corectate pt. cei " << n_detectors << " detectori:" << endl; }
        cout << "C_exp[" << i+1 << "] = "; cin >> X_exp[i];
    }

    for(int i=0; i<n_samples; i++)
    {
        double sum = 0;

        for(int j=0; j<n_detectors; j++)
        {
            sum += (X_exp[j] - C[i][j]) * (X_exp[j] - C[i][j]);
        }
        d[i] = sqrt(sum);
        w[i] = gaussian(d[i], sigma);
    }

    for(int i=0; i<n_energy_groups; i++)
    {
        double numarator = 0;
        double numitor = 0;

        for(int j=0; j<n_samples; j++)
        {
            numarator += w[j] * phi[j][i];
            numitor += w[j];
        }
        y_exp[i] = numarator / numitor;
    }

    // Salvare spectru calculat

    ofile.open("spectru_exp_prezis.txt");
    for(int i=0; i<n_energy_groups; i++)
    {
        if(i == 0) { ofile << "Energy group Eg  |  Phi(Eg)" << endl; }
        ofile << i+1 << " " << y_exp[i] << endl;
    }
    ofile.close();
    return 0;
}
