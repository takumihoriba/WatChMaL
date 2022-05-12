#include <stdio.h>
#include <math.h>

#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include "loweroot.h"
#include "tqrealroot.h"

int main(int argc,char **argv)
{
TFile      *in,*out;
TTree      *inwit,*outwit;
LoweInfo   *lowe;
TQReal     *tqreal;
TBranch    *lowebranch;

in=new TFile(argv[1]);
inwit=(TTree *) in->Get("data");
tqreal=new TQReal;
lowe=new LoweInfo;
inwit->SetBranchAddress("TQREAL",&tqreal);
inwit->SetBranchAddress("LOWE",&lowe,&lowebranch);

out=new TFile(argv[2],"RECREATE");
if (out==NULL) return(0);
out->cd();
outwit=new TTree("data","Prepreprocess SKROOT");
outwit->Branch("LOWE",&lowe);

outwit->Branch("TQREAL",&tqreal);

printf("%d\n",inwit->GetEntries());
for(int entry=0;entry<inwit->GetEntries();entry++)
{
    if (entry % 10000 == 0) {printf("%d\n",entry);}
    inwit->GetEntry(entry);
    outwit->Fill();
}
    
in->Close();
out->cd();
outwit->Write();
out->Close();  
}