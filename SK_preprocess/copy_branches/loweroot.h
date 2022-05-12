#ifndef LOWEROOT_H
#define LOWEROOT_H

#include "TObject.h"
#include "TNamed.h"
#include <vector>

// Maximilien Fechner, aout 2007

// in this file we define the classes that
// contain low-energy realted information

// 30-AUG-2007 modified by Y.Takeuchi 
//   o modified for lowe 
// 03-OCT-2007 modified by Y.Takeuchi 
//   o added MuInfo
// 15-OCT-2007 modified by Y.Takeuchi 
//   o added SLEInfo
// 20-OCT-2007 modified by Y.Takeuchi 
//   o added patlik, lwatert, linfo[]
// 09-MAY-2008 modified by Y.Takeuchi 
//   o added spadll spadl -> spadlt, spaevnum (LoweInfo, 2)
//   o added muboy info. (MuInfo, 2)
// 01-OCT-2008 modified by Y.Takeuchi 
//   o added bff parameters (MuInfo, 4)
//   o modified SLE for SK-IV (SLEInfo, 3)

class LoweInfo : public TNamed {
public:
    Float_t  bsvertex[4];  // bonsai vertex:  x, y, z, t
    Float_t  bsresult[6];  // bonsai results: theta, phi, alpha, cos_theta, epsilon, like_t
    Float_t  bsdir[3];     // bonsai direction(x,y,z) based on bonsai vertex
    Float_t  bsgood[3];    // bonsai goodness: likelihood, goodness, ?
    Float_t  bsdirks;      // bonsai direction KS 
    Float_t  bseffhit[12]; // bonsai effective hits at fixed water transparencies
    Float_t  bsenergy;     // bonsai energy
    Int_t    bsn50;        // bonsai # of hits in 50 nsec after TOF subtraction
    Float_t  bscossun;     // bonsai cossun
    Float_t  clvertex[4];  // clusfit vertex: x, y, z, t
    Float_t  clresult[4];  // clusfit results: theta, phi, cos_theta, good_t
    Float_t  cldir[3];     // clusfit direction(x,y,z) based on clusfit vertex
    Float_t  clgoodness;   // clusfit goodness 
    Float_t  cldirks;      // clusfit direction KS
    Float_t  cleffhit[12]; // clusfit effective hits at fixed water transparencies
    Float_t  clenergy;     // clusfit energy
    Int_t    cln50;        // clusfit # of hits in 50 nsec after TOF subtraction
    Float_t  clcossun;     // clusfit cossun
    Int_t    latmnum;      // ATM number
    Int_t    latmh;        // ATM hit
    Int_t    lmx24;        // max 24
    Double_t ltimediff;    // time to the previous LE event (in raw data)
    Float_t  lnsratio;     // Noise-Signal ratio
    Float_t  lsdir[3];     // solar direction at the time (x,y,z)
    Int_t    spaevnum;     // event number of the parent muon 
    Float_t  spaloglike;   // spallation log likelihood
    Float_t  sparesq;      // spallation residual charge
    Float_t  spadt;        // spallation delta-T between parent muon
    Float_t  spadll;       // longitudinal distance
    Float_t  spadlt;       // traversal distance 
    Int_t    spamuyn;      // spallation muyn
    Float_t  spamugdn;     // spallation muon goodness
    Float_t  posmc[3];     // MC true vertex position
    Float_t  dirmc[2][3];  // MC true direction (1st and 2nd particles)
    Float_t  pabsmc[2];    // MC absolute momentum (1st & 2nd)
    Float_t  energymc[2];  // MC generated energy(s) (1st & 2nd)
    Float_t  darkmc;       // MC dark rate for generation
    Int_t    islekeep;     // SLE keep flag
// below is added on 30-OCT-2007 y.t. (still version 1)
    Float_t  bspatlik;     // bonsai pattern likelihood
    Float_t  clpatlik;     // bonsai pattern likelihood
    Float_t  lwatert;      // water transparency value (at reconstruction?)
    Int_t    lninfo;       // # of extra low-e inforamation
    Int_t    linfo[255];   // extra low-e information (see skroot_lowe.h)

    void Clear() {
	for (int i = 0; i < 4; i++) 
	    bsvertex[i] = 0.;
	for (int i = 0; i < 6; i++) 
	    bsresult[i] = 0.;
	for (int i = 0; i < 3; i++) 
	    bsdir[i] = 0.;
	for (int i = 0; i < 3; i++) 
	    bsgood[i] = 0.;
	bsdirks = 0.;
	for (int i = 0; i < 12; i++) 
	    bseffhit[i] = 0.;
	bsenergy = 0.;
	bsn50 = 0;
	bscossun = 0.;
	for (int i = 0; i < 4; i++) 
	    clvertex[i] = 0.;
	for (int i = 0; i < 4; i++) 
	    clresult[i] = 0.;
	for (int i = 0; i < 3; i++) 
	    cldir[i] = 0.;
	clgoodness = 0.;
	cldirks = 0.;
	for (int i = 0; i < 12; i++) 
	    cleffhit[i] = 0.;
	clenergy = 0.;
	cln50 = 0;
	clcossun = 0.;
	latmnum = 0;
	latmh = 0;
	lmx24 = 0;
	ltimediff = 0.;
	lnsratio = 0.;
	for (int i = 0; i < 3; i++) 
	    lsdir[i] = 0.;
	spaevnum = 0;
	spaloglike = 0.;
	sparesq = 0.;
	spadt = 0.;
	spadll = 0.;
	spadlt = 0.;
	spamuyn = 0;
	spamugdn = 0.;
	for (int i = 0; i < 3; i++) 
	    posmc[i] = 0.;
	for (int j = 0; j < 3; j++) 
	    for (int i = 0; i < 2; i++) 
		dirmc[i][j] = 0.;
	for (int i = 0; i < 2; i++) 
	    pabsmc[i] = 0.;
	for (int i = 0; i < 2; i++) 
	    energymc[i] = 0.;
	darkmc = 0.;
	islekeep = 0;
	bspatlik = 0.0;
	clpatlik = 0.0;
	lwatert = 0.0;
        lninfo = 0;
	for (int i = 0; i < 255; i++) 
	    linfo[i] = 0;
    }
  
    ClassDef(LoweInfo,3) // increase version number when structure changed. 
};


class MuInfo : public TNamed {
public:
    Float_t  muentpoint[3];
    Float_t  mudir[3];
    Double_t mutimediff;
    Float_t  mugoodness;
    Float_t  muqismsk;
    Int_t    muyn;
    Int_t    mufast_flag;
    Int_t    muboy_status;         // muboy status
    Int_t    muboy_ntrack;         // number of tracks
    Float_t  muboy_entpos[10][4];  // up to 10 tracks
    Float_t  muboy_dir[3];         // common direction
    Float_t  muboy_goodness;       // goodness
    Float_t  muboy_length;         // track length
    Float_t  muboy_dedx[200];      // dE/dX histogram
    Float_t  mubff_entpos[3];      // bff entpos
    Float_t  mubff_dir[3];         // bff direction
    Float_t  mubff_goodness;       // bff goodness
    Int_t    muninfo;              // number of additional data in muinfo
    Int_t    muinfo[255];          // additional data

    void Clear() {
	for (int i = 0; i < 3; i++) 
	    muentpoint[i] = 0.;
	for (int i = 0; i < 3; i++) 
	    mudir[i] = 0.;
	mutimediff = 0.0;
	mugoodness = 0.0;
	muqismsk = 0.0;
	muyn = 0;
	mufast_flag = 0;

	muboy_status = 0;
	muboy_ntrack = 0;
	for (int i = 0; i < 10; i++)
	    for (int j = 0; j < 4; j++) {
		muboy_entpos[i][j] = 0.;
	    }
	for (int i = 0; i < 3; i++)
	    muboy_dir[i] = 0.;
	muboy_goodness = 0.;
	muboy_length = 0.;
	for (int i = 0; i < 200; i++)
	muboy_dedx[i] = 0.;

	for (int i = 0; i < 3; i++) {
	    mubff_entpos[i] = 0.;
	    mubff_dir[i] = 0.;
	}
	mubff_goodness = 0.;

        muninfo = 0;
	for (int i = 0; i < 255; i++) 
	    muinfo[i] = 0;
    }
  
    ClassDef(MuInfo,4)
};

class SLEInfo : public TNamed {
public:

    Float_t  wallcut;
    Int_t    nsel;

    Float_t  itbsvertex[4];
    Float_t  itbsresult[6];
    Float_t  itbsgood[3];
    Int_t    nbonsai;

    Float_t  itcfvertex[4];
    Float_t  itcfresult[4];
    Float_t  itcfgoodness;
    Int_t    nclusfit;
 
    void Clear() {

        wallcut=0.;
        nsel=0;
	for (int i = 0; i < 4; i++) 
	    itbsvertex[i] = 0.;
	for (int i = 0; i < 6; i++) 
	    itbsresult[i] = 0.;
	for (int i = 0; i < 3; i++) 
	    itbsgood[i] = 0.;
	nbonsai=0;

	for (int i = 0; i < 4; i++) 
	    itcfvertex[i] = 0.;
	for (int i = 0; i < 4; i++) 
	    itcfresult[i] = 0.;
	itcfgoodness = 0.;
	nclusfit=0;
    }
    ClassDef(SLEInfo,3)
};

#endif

