import numpy as np
import os
import argparse
import h5py
import root_utils.pos_utils as pu

def get_args():
    parser = argparse.ArgumentParser(description='convert and merge .npz files to hdf5')
    parser.add_argument('input_files', type=str, nargs='+')
    parser.add_argument('-o', '--output_file', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    config = get_args()
    print("ouput file:", config.output_file)
    f = h5py.File(config.output_file, 'w')

    total_rows = 0
    total_hits = 0
    min_hits = 1
    good_rows = 0
    good_hits = 0
    print("counting events and hits, in files")
    file_event_hits = {}
    file_good_events = {}
    for input_file in config.input_files:
        print(input_file)
        if not os.path.isfile(input_file):
            raise ValueError(input_file+" does not exist")
        npz_file = np.load(input_file)
        trigger_time = npz_file['trigger_time']
        hit_trigger = npz_file['digi_hit_trigger']
        total_rows += hit_trigger.shape[0]
        event_hits = []
        good_events = []
        for i in range(hit_trigger.shape[0]):
            first_trigger = np.argmin(trigger_time[i])
            hits = np.where(hit_trigger[i]==first_trigger)
            nhits = len(hits[0])
            total_hits += nhits
            if nhits >= min_hits:
                good_events.append(i)
                event_hits.append(hits)
                good_hits += nhits
                good_rows += 1
        file_event_hits[input_file] = event_hits
        file_good_events[input_file] = good_events
    
    print(len(config.input_files), "files with", total_rows, "events with ", total_hits, "hits")
    print(good_rows, "events with at least", min_hits, "hits for a total of", good_hits, "hits")

    dset_labels=f.create_dataset("labels",
                                 shape=(good_rows,),
                                 dtype=np.int32)
    dset_PATHS=f.create_dataset("root_files",
                                shape=(good_rows,),
                                dtype=h5py.special_dtype(vlen=str))
    dset_IDX=f.create_dataset("event_ids",
                              shape=(good_rows,),
                              dtype=np.int32)
    dset_hit_time=f.create_dataset("hit_time",
                                 shape=(good_hits, ),
                                 dtype=np.float32)
    dset_hit_charge=f.create_dataset("hit_charge",
                                 shape=(good_hits, ),
                                 dtype=np.float32)
    dset_hit_pmt=f.create_dataset("hit_pmt",
                                  shape=(good_hits, ),
                                  dtype=np.int16)
    dset_event_hit_index=f.create_dataset("event_hits_index",
                                          shape=(good_rows,),
                                          dtype=np.int64)
    dset_energies=f.create_dataset("energies",
                                   shape=(good_rows, 1),
                                   dtype=np.float32)
    dset_positions=f.create_dataset("positions",
                                    shape=(good_rows, 1, 3),
                                    dtype=np.float32)
    dset_angles=f.create_dataset("angles",
                                 shape=(good_rows, 2),
                                 dtype=np.float32)
    offset = 0
    offset_next = 0
    hit_offset = 0
    hit_offset_next = 0
    label_map = {22: 0, 11: 1, 13: 2}
    for input_file in config.input_files:
        print(input_file)
        npz_file = np.load(input_file, allow_pickle=True)
        good_events = file_good_events[input_file]
        hit_pmt = npz_file['digi_hit_pmt'][good_events]
        event_id = npz_file['event_id'][good_events]
        root_file = npz_file['root_file'][good_events]
        pid = npz_file['pid'][good_events]
        position = npz_file['position'][good_events]
        direction = npz_file['direction'][good_events]
        energy = npz_file['energy'][good_events]
        hit_time = npz_file['digi_hit_time'][good_events]
        hit_charge = npz_file['digi_hit_charge'][good_events]
        hit_pmt = npz_file['digi_hit_pmt'][good_events]

        offset_next += event_id.shape[0]

        dset_IDX[offset:offset_next] = event_id
        dset_PATHS[offset:offset_next] = root_file
        dset_energies[offset:offset_next,:] = energy.reshape(-1,1)
        dset_positions[offset:offset_next,:,:] = position.reshape(-1,1,3)

        labels = np.full(pid.shape[0], -1)
        for l, v in label_map.items():
            labels[pid==l] = v
        dset_labels[offset:offset_next] = labels

        polar = np.arccos(direction[:,1])
        azimuth = np.arctan2(direction[:,2], direction[:,0])
        dset_angles[offset:offset_next,:] = np.hstack((polar.reshape(-1,1),azimuth.reshape(-1,1)))

        for i in range(hit_pmt.shape[0]):
            dset_event_hit_index[offset+i] = hit_offset
            hit_indices = file_event_hits[input_file][i]
            hit_offset_next += len(hit_indices[0])
            dset_hit_time[hit_offset:hit_offset_next] = hit_time[i][hit_indices]
            dset_hit_charge[hit_offset:hit_offset_next] = hit_charge[i][hit_indices]
            dset_hit_pmt[hit_offset:hit_offset_next] = hit_pmt[i][hit_indices]
            hit_offset = hit_offset_next

        offset = offset_next
    f.close()
    print("saved", hit_offset, "hits in", offset, "good events (each with at least", min_hits, "hits)")
