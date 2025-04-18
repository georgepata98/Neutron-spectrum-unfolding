// ============================================================
// Spectrele trebuie sa aiba 2 coloane, y_test respectiv y_pred
// ============================================================
{
    TCanvas *c1 = new TCanvas("c1", "Canvas", 800, 600);
    TLegend *legend = new TLegend(.6, .7, .9, .9);
    TGraph *gr1 = new TGraph();
    TGraph *gr2 = new TGraph();
    TMultiGraph *mg = new TMultiGraph();

    ifstream ifile;
    string filename, firstline;

    cout << "Nume fisier cu spectre (fara .txt): "; cin >> filename;

    ifile.open((filename + ".txt").c_str());
    if(!ifile) { cout << filename << ".txt nu exista." << endl; }
    if(ifile)
    {
        double y_test, y_pred;
        int i = 0;

        getline(ifile, firstline);  // ia prima linie cu numele coloanelor (pentru a sari peste ea in loop-ul while)
        while(1)
        {
            ifile >> y_test >> y_pred;
            if(ifile.eof()) break;
            gr1->SetPoint(i, i, y_test);
            gr2->SetPoint(i, i, y_pred);
            i++;
        }
        ifile.close();
    }


    // --- Plotare ---
    c1->SetGrid();
    gr1->SetLineWidth(2);
    gr2->SetLineWidth(2);
    gr1->SetLineColor(kRed);
    gr2->SetLineColor(kBlue);
    mg->Add(gr1);
    mg->Add(gr2);
    mg->SetTitle("True vs. Unfolded Spectrum");
    mg->GetXaxis()->SetTitle("Energy Group");
    mg->GetYaxis()->SetTitle("Fluence [a.u.]");
    legend->AddEntry(gr1, "true spectrum");
    legend->AddEntry(gr2, "GRNN predicted spectrum");
    mg->Draw("AL");
    legend->Draw("SAME");
    c1->Update();
}
