// Plotare MSE functie de sigma
{
    TGraph* gr1 = new TGraph();

    ifstream ifile;
    string filename;

    //cout << "Nume fisier cu MSE(sigma) (fara .txt): "; cin >> filename;
    filename = "mse_data";

    ifile.open((filename + ".txt").c_str());
    if(!ifile) { cout << filename << ".txt nu exista!" << endl; }
    if(ifile)
    {
        double mse, sigma;
        int i = 0;

        while(1)
        {
            ifile >> mse >> sigma;
            if(ifile.eof()) break;
            gr1->SetPoint(i, sigma, mse);
            i++;
        }
        ifile.close();


        // --- Plotare ---
        TCanvas *c1 = new TCanvas("c1", "Canvas", 800, 600);
        c1->SetGrid();
        gr1->SetLineWidth(2);
        gr1->SetLineColor(kBlue);
        gr1->SetTitle("MSE vs. sigma");
        gr1->GetXaxis()->SetTitle("sigma");
        gr1->GetYaxis()->SetTitle("MSE");
        gr1->Draw("AL");
        c1->Update();
    }
}
