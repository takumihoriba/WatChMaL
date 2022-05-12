#include "tqrealroot.h"
#include <iostream>

ClassImp(Header)
ClassImp(TQReal)

//-------------------------------------------------
void Header::Clear()
{
    nrunsk = 0;
    nsubsk = 0;
    nevsk = 0;
    for (int i = 0; i < 3; i++) 
	ndaysk[i] = 0;
    for (int i = 0; i < 4; i++) 
	ntimsk[i] = 0;
    for (int i = 0; i < 3; i++) 
	nt48sk[i] = 0;
    mdrnsk = 0;
    idtgsk = 0;
    ifevsk = 0;
    ltcgps = 0;
    nsgps = 0; 
    nusgps = 0; 
    ltctrg = 0; 
    ltcbip = 0;
    iffscc = 0;
    for (int i = 0; i < 4; i++) 
	itdct0[i] = 0;
    icalva = 0;
    sk_geometry = 0;  
    
    onldata_block_id = 0;
    swtrg_id = 0;
    counter_32 = 0;
    hostid_high = 0;
    hostid_low = 0;
    t0 = 0;
    gate_width = 0;
    sw_trg_mask = 0;
    contents = 0;
    ntrg = 0;
};

//-------------------------------------------------
// dump variables as a test
//-------------------------------------------------
void Header::Print()
{
    std::cout << " " << nrunsk
	      << " " << nsubsk 
	      << " " << nevsk
	      << " " << ndaysk[0]
	      << " " << ndaysk[1]
	      << " " << ndaysk[2]
	      << " " << ntimsk[0]
	      << " " << ntimsk[1]
	      << " " << ntimsk[2]
	      << " " << ntimsk[3]
	      << " " << nt48sk[0]
	      << " " << nt48sk[1]
	      << " " << nt48sk[2]
	      << " " << mdrnsk
	      << " " << idtgsk
	      << " " << ifevsk
	      << " " << ltcgps
	      << " " << nsgps
	      << " " << ltctrg
	      << " " << ltcbip
	      << " " << iffscc
	      << " " << itdct0[0]
	      << " " << itdct0[1]
	      << " " << itdct0[2]
	      << " " << itdct0[3]
	      << " " << icalva
	      << " " << sk_geometry
	      << " " << onldata_block_id
	      << " " << swtrg_id
	      << " " << counter_32
	      << " " << hostid_high
	      << " " << hostid_low
	      << " " << t0
	      << " " << gate_width
	      << " " << sw_trg_mask
	      << " " << contents
	      << " " << ntrg
	      << std::endl;
};


//-------------------------------------------------
void TQReal::Clear()
{
  nhits = 0;
  pc2pe = 0.0;
  tqreal_version = 0;
  qbconst_version = 0;
  tqmap_version = 0;
  pgain_version = 0;
  it0xsk = 0;
  // very important to clear the vector between entries
  cables.clear();
  T.clear();
  Q.clear();  
}
