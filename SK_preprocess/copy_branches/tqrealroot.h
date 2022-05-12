#ifndef TQREAL_DEFINITION_H
#define TQREAL_DEFINITION_H

#define MAXPM 11146
#define MAXPMA 1885

#include "TObject.h"
#include "TNamed.h"
#include <vector>

class Header : public TNamed {
public:
  
    Int_t nrunsk;
    Int_t nsubsk;
    Int_t nevsk;
    Int_t ndaysk[3];
    Int_t ntimsk[4];
    Int_t nt48sk[3];
    Int_t mdrnsk;
    Int_t idtgsk;
    Int_t ifevsk;
    Int_t ltcgps;
    Int_t nsgps; 
    Int_t nusgps; 
    Int_t ltctrg; 
    Int_t ltcbip;
    Int_t iffscc;
    Int_t itdct0[4];
    Int_t icalva;
    Int_t sk_geometry;  

    // for SK-IV
    Int_t onldata_block_id;  // block id in online data format
    Int_t swtrg_id;          // software trigger ID 
    Int_t counter_32;        //   
    Int_t hostid_high;       // 
    Int_t hostid_low;        // 
    Int_t t0;                // t0 from software trigger
    Int_t gate_width;        // gate width from software trigger
    Int_t sw_trg_mask;       // 
    Int_t contents;          //  
    Int_t ntrg;              // number of hardware trigger from ST
  
    void Clear();
    void Print();

    ClassDef(Header,3)
};

class TQReal : public TNamed {
public:
  
  Int_t nhits;
  Float_t pc2pe;
  Int_t tqreal_version;  // added in ver.2
  Int_t qbconst_version; // added in ver.2
  Int_t tqmap_version;   // added in ver.2
  Int_t pgain_version;   // added in ver.2
  Int_t it0xsk;          // added in ver.2

  std::vector<int> cables;
  std::vector<float> T;
  std::vector<float> Q;

  void Clear();

  ClassDef(TQReal,3)

};

#endif
