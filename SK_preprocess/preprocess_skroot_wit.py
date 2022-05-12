'''
Example preprocssing script. Reads in SKRoot file for signal and WIT file for background.
Creates h5 file in WatChMaL format and npz file for train-val-test indices.
The SKRoot file must first be copied with the copy_branches program.
'''

import uproot3
import numpy as np
import h5py

fsig = uproot3.open('/Users/Alejandro/Desktop/skdetsim.b8.detsim_rdir.r062361.r077336.prepreprocess.root')
fbg = uproot3.open('/Users/Alejandro/Documents/mcwit/data/redwit.074129.077889.lowfitwitE.root')
fout = h5py.File('SKROOT_B8_redwit_4MeV.h5','w')
findex = 'SKROOT_B8_redwit_4MeV_idxs.npz'

treesig=fsig['data']
treebg=fbg['wit']

nhitsig = treesig.array("nhits")
nhitbg = treebg.array("nhit")

nsig=len(nhitsig)
nbg=len(nhitbg)

hits_index_sig = np.append(0,np.cumsum(nhitsig)[:-1])
hits_index_bg = np.append(0,np.cumsum(nhitbg)[:-1])+hits_index_sig[-1]+nhitsig[-1]

fout.create_dataset("labels",data=np.append(np.ones(nsig,dtype="i4"),np.zeros(nbg,dtype="i4")))
fout.create_dataset("event_hits_index", data=np.append(hits_index_sig, hits_index_bg))
fout.create_dataset("nhit",data=np.append(nhitsig, nhitbg))
fout.create_dataset("hit_pmt",data=np.append(np.bitwise_and(treesig.array("cables").flatten(),int(0xffff)),treebg.array("cable").flatten()),dtype='i4')
#fout.create_dataset("hit_time",data=np.append(treesig.array("T").flatten(),treebg.array("t").flatten()))
fout.create_dataset("hit_charge",data=np.append(treesig.array("Q").flatten(),treebg.array("q").flatten()))

times_sig = treesig.array("T").tolist()
times_bg = treebg.array("t").tolist()
for iEvt in range(nsig):
    times_sig[iEvt] = np.array(times_sig[iEvt]) - (times_sig[iEvt][0] + times_sig[iEvt][-1])/2
for iEvt in range(nbg):  
    times_bg[iEvt] = np.array(times_bg[iEvt]) - (times_bg[iEvt][0] + times_bg[iEvt][-1])/2

times_sig = np.array([t for evt in times_sig for t in evt])
times_bg = np.array([t for evt in times_bg for t in evt])
fout.create_dataset("hit_time",data=np.append(times_sig,times_bg),dtype='f4')

fout.close()

ntrain = int(0.8*(nsig+nbg))
nval = int(0.1*(nsig+nbg))

shuffle_idxs = np.arange(nsig+nbg)
np.random.shuffle(shuffle_idxs)

train_idxs = shuffle_idxs[:ntrain]
val_idxs = shuffle_idxs[ntrain:ntrain+nval]
test_idxs = shuffle_idxs[ntrain+nval:]

np.savez(findex, train_idxs=train_idxs, val_idxs=val_idxs, test_idxs=test_idxs)

#np.savez(findex, test_idxs = np.arange(2*nsig))


fsig.close()
fbg.close()