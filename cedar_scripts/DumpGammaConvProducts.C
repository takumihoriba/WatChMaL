/*
 * Dump gamma conversion products to nuance txt file for simulation at desired position / direction
 * 
 * if dir is "2pi" then direction is randomly rotated about tank axis
 * if dir is "4pi" then direction is rotated to random isotropic
 * otherwise dir is left as is
 * if pos is "unif" then position is uniform random in cylinder with radius x and half-length y
 * otherwise pos is set to (x, y, z)
 *
 */
#include <iostream>
#include <fstream>
#include <TFile.h>
#include <TTree.h>
#include <TRandom3.h>
#include <TMath.h>
#include <TVector3.h>

using namespace std;

void DumpGammaConvProducts(char * infile, char * outfile, int total, char * dir, char * pos, double y, double x, double z=0){
  TRandom3 *r = new TRandom3(0);
  TFile * f = new TFile(infile);
  TTree * t = (TTree*) f->Get("Tracks");
  int ntrack, pid[1000], parentid[1000], trackid[1000];
  float dirx[1000], diry[1000], dirz[1000], p[1000], m[1000], startx[1000], starty[1000], startz[1000], stopx[1000], stopy[1000], stopz[1000];
  t->SetBranchAddress("Pid",pid);
  t->SetBranchAddress("Ntracks",&ntrack);
  t->SetBranchAddress("ParentID",parentid);
  t->SetBranchAddress("TrackID",trackid);
  t->SetBranchAddress("Dirx",dirx);
  t->SetBranchAddress("Diry",diry);
  t->SetBranchAddress("Dirz",dirz);
  t->SetBranchAddress("Start_x",startx);
  t->SetBranchAddress("Start_y",starty);
  t->SetBranchAddress("Start_z",startz);
  t->SetBranchAddress("Stop_x",stopx);
  t->SetBranchAddress("Stop_y",stopy);
  t->SetBranchAddress("Stop_z",stopz);
  t->SetBranchAddress("P",p);
  t->SetBranchAddress("Mass",m);
  int n=t->GetEntries();
  int N=0;
  std::ofstream out(outfile,std::ofstream::out);
  double maxy = y, maxR = x;
  for(int i=0; i<n; i++){
    t->GetEntry(i);
    int gamma=-1;
    for(int it=0; it<ntrack; it++){
      if(parentid[it]==0 && pid[it]==22){
        gamma=it;
        break;
      }
    }
    if(gamma<0){
      cout << "WARNING: event with no gamma found, skipping" << endl;
      continue;
    }
//    if(stopy[gamma]!=0 || stopz[gamma]!=0){
//      cout << "WARNING: gamma didn't move only in x dir, skipping" << endl;
//      continue;
//    }
    int electron=-1, positron=-1;
    for(int it=0; it<ntrack; it++){
      if(parentid[it]==trackid[gamma] && startx[it]==stopx[gamma]){
        if(pid[it]==11 ) electron=it;
        else if(pid[it]==-11) positron=it;
      }
    }
    if(electron==-1 || positron==-1){
//      cout << "WARNING: no e+/e- pair from gamma conversion found" << endl;
      continue;
    }
    float electronE = sqrt(p[electron]*p[electron]+m[electron]*m[electron]);
    float positronE = sqrt(p[positron]*p[positron]+m[positron]*m[positron]);
    if(electronE+positronE < p[gamma]-1.0){
//      cout << "WARNING: electron and positron lower than expected energy, skipping" << endl;
      continue;
    }
    if(!strcmp(pos, "unif")) {
      double R=maxR*TMath::Sqrt(r->Uniform());
      r->Circle(x,z,R);
      y=r->Uniform(-maxy, maxy);
    }
    if(!strcmp(dir, "2pi")){
      double angle = r->Uniform(TMath::Pi()*2);
      TVector3 gammaDir(dirx[gamma], diry[gamma], dirz[gamma]);
      TVector3 electronDir(dirx[electron], diry[electron], dirz[electron]);
      TVector3 positronDir(dirx[positron], diry[positron], dirz[positron]);
      gammaDir.RotateY(angle);
      electronDir.RotateY(angle);
      positronDir.RotateY(angle);
      dirx[gamma] = gammaDir.X();
      dirz[gamma] = gammaDir.Z();
      dirx[electron] = electronDir.X();
      dirz[electron] = electronDir.Z();
      dirx[positron] = positronDir.X();
      dirz[positron] = positronDir.Z();
    }
    else if (!strcmp(dir, "4pi")){
      TVector3 gammaDir(dirx[gamma], diry[gamma], dirz[gamma]);
      TVector3 electronDir(dirx[electron], diry[electron], dirz[electron]);
      TVector3 positronDir(dirx[positron], diry[positron], dirz[positron]);
      // I think this performs a uniform random rotation... it's a harder problem than first appears
      // First rotate x axis to some isotropic random direction
      double rx, ry, rz;
      r->Sphere(rx, ry, rz, 1);
      TVector3 randDir(rx,ry,rz);
      TVector3 xDir(1,0,0);
      TVector3 axis = xDir.Cross(randDir);
      double angle = xDir.Angle(randDir);
      gammaDir.Rotate(angle, axis);
      electronDir.Rotate(angle, axis);
      positronDir.Rotate(angle, axis);
      // Now rotate uniform random angle about the random direction
      angle = r->Uniform(TMath::Pi()*2);
      gammaDir.Rotate(angle, randDir);
      electronDir.Rotate(angle, randDir);
      positronDir.Rotate(angle, randDir);
      dirx[gamma] = gammaDir.X();
      diry[gamma] = gammaDir.Y();
      dirz[gamma] = gammaDir.Z();
      dirx[electron] = electronDir.X();
      diry[electron] = electronDir.Y();
      dirz[electron] = electronDir.Z();
      dirx[positron] = positronDir.X();
      diry[positron] = positronDir.Y();
      dirz[positron] = positronDir.Z();
    }
    out << "$ begin" << endl;
    out << "$ nuance " << 0 << endl;
    out << "$ vertex " << x << " " << y << " " << z << " 0" << endl;
    out << "$ track 22 " << p[gamma] << " " << dirx[gamma] << " " << diry[gamma] << " " << dirz[gamma] << " -1" << endl; // Initial gamma
    out << "$ track 2212 0 0 0 0 -1" << endl; // dummy target
    out << "$ info 0 " << N << endl;
    out << "$ track 11 " << electronE << " " << dirx[electron] << " " << diry[electron] << " " << dirz[electron] << " 0" << endl;
    out << "$ track -11 " << positronE << " " << dirx[positron] << " " << diry[positron] << " " << dirz[positron] << " 0" << endl;
    out << "$ end" << endl;
    N++;
    if(N==total) break;
  }
  if(N!=total) cout << "WARNING: " << N << " gamma conversions is less than " << total << "!" << endl;
  out.close();
}
