import numpy as np
import os
import sys
import subprocess
from datetime import datetime
import argparse
import h5py
import root_utils.pos_utils as pu

def get_args():
    parser = argparse.ArgumentParser(description='convert and merge .npz files to hdf5')
    parser.add_argument('input_files', type=str, nargs='+')
    parser.add_argument('-o', '--output_file', type=str)
    parser.add_argument('-H', '--half-height', type=float, default=300)
    parser.add_argument('-R', '--radius', type=float, default=400)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    config = get_args()
    print("ouput file:", config.output_file)
    f = h5py.File(config.output_file, 'w')
    
    script_path = os.path.dirname(os.path.abspath(__file__))
    git_status = subprocess.check_output(['git', '-C', script_path, 'status', '--porcelain', '--untracked-files=no']).decode()
    if git_status:
        raise Exception("Directory of this script ({}) is not a clean git directory:\n{}Need a clean git directory for storing script version in output file.".format(script_path, git_status))
    git_describe = subprocess.check_output(['git', '-C', script_path, 'describe', '--always', '--long', '--tags']).decode().strip()
    print("git describe for path to this script ({}):".format(script_path), git_describe)
    f.attrs['git-describe'] = git_describe
    f.attrs['command'] = str(sys.argv)
    f.attrs['timestamp'] = str(datetime.now())

    total_rows = 0
    total_hits = 0
    print("counting events and hits in files")
    for input_file in config.input_files:
        print(input_file, flush=True)
        if not os.path.isfile(input_file):
            raise ValueError(input_file+" does not exist")
        npz_file = np.load(input_file)
        hit_pmts = npz_file['true_hit_pmt']
        total_rows += hit_pmts.shape[0]
        for h in hit_pmts:
            total_hits += h.shape[0]
    
    print(len(config.input_files), "files with", total_rows, "events with ", total_hits, "hits")

    dset_labels=f.create_dataset("labels",
                                 shape=(total_rows,),
                                 dtype=np.int32)
    dset_PATHS=f.create_dataset("root_files",
                                shape=(total_rows,),
                                dtype=h5py.special_dtype(vlen=str))
    dset_IDX=f.create_dataset("event_ids",
                              shape=(total_rows,),
                              dtype=np.int32)
    dset_hit_time=f.create_dataset("hit_time",
                                 shape=(total_hits, ),
                                 dtype=np.float32)
    dset_hit_pmt=f.create_dataset("hit_pmt",
                                  shape=(total_hits, ),
                                  dtype=np.int32)
    dset_event_hit_index=f.create_dataset("event_hits_index",
                                          shape=(total_rows,),
                                          dtype=np.int64) # int32 is too small to fit large indices
    dset_energies=f.create_dataset("energies",
                                   shape=(total_rows, 1),
                                   dtype=np.float32)
    dset_positions=f.create_dataset("positions",
                                    shape=(total_rows, 1, 3),
                                    dtype=np.float32)
    dset_angles=f.create_dataset("angles",
                                 shape=(total_rows, 2),
                                 dtype=np.float32)
    dset_veto = f.create_dataset("veto",
                                 shape=(total_rows,),
                                 dtype=np.bool_)
    dset_veto2 = f.create_dataset("veto2",
                                  shape=(total_rows,),
                                  dtype=np.bool_)

    offset = 0
    offset_next = 0
    hit_offset = 0
    hit_offset_next = 0
    label_map = {22: 0, 11: 1, 13: 2}
    for input_file in config.input_files:
        print(input_file, flush=True)
        npz_file = np.load(input_file, allow_pickle=True)
        event_ids = npz_file['event_id']
        root_files = npz_file['root_file']
        pids = npz_file['pid']
        positions = npz_file['position']
        directions = npz_file['direction']
        energies = npz_file['energy']
        hit_times = npz_file['true_hit_time']
        hit_pmts = npz_file['true_hit_pmt']
        hit_triggers = npz_file['digi_hit_trigger']
        track_pid = npz_file['track_pid']
        track_energy = npz_file['track_energy']
        track_stop_position = npz_file['track_stop_position']
        track_start_position = npz_file['track_start_position']


        offset_next += event_ids.shape[0]

        dset_IDX[offset:offset_next] = event_ids
        dset_PATHS[offset:offset_next] = root_files
        dset_energies[offset:offset_next,:] = energies.reshape(-1,1)
        dset_positions[offset:offset_next,:,:] = positions.reshape(-1,1,3)

        labels = np.full(pids.shape[0], -1)
        for l, v in label_map.items():
            labels[pids==l] = v
        dset_labels[offset:offset_next] = labels

        polars = np.arccos(directions[:,1])
        azimuths = np.arctan2(directions[:,2], directions[:,0])
        dset_angles[offset:offset_next,:] = np.hstack((polars.reshape(-1,1),azimuths.reshape(-1,1)))

        for i, (pids, energies, starts, stops) in enumerate(zip(track_pid, track_energy,track_start_position, track_stop_position)):
            muons_above_threshold = (np.abs(pids) == 13) & (energies > 166)
            electrons_above_threshold = (np.abs(pids) == 11) & (energies > 2)
            gammas_above_threshold = (np.abs(pids) == 22) & (energies > 2)
            above_threshold = muons_above_threshold | electrons_above_threshold | gammas_above_threshold
            outside_tank = (np.linalg.norm(stops[:,(0,2)], axis=1) > config.radius) | (np.abs(stops[:, 1]) > config.half_height)
            dset_veto[offset+i] = np.any(above_threshold & outside_tank)
            end_energy_estimate = energies - np.linalg.norm(stops - starts)*2
            muons_above_threshold = (np.abs(pids) == 13) & (end_energy_estimate > 166)
            electrons_above_threshold = (np.abs(pids) == 11) & (end_energy_estimate > 2)
            gammas_above_threshold = (np.abs(pids) == 22) & (end_energy_estimate > 2)
            above_threshold = muons_above_threshold | electrons_above_threshold | gammas_above_threshold
            dset_veto2[offset+i] = np.any(above_threshold & outside_tank)

        for i, (times, pmts) in enumerate(zip(hit_times, hit_pmts)):
            dset_event_hit_index[offset+i] = hit_offset
            hit_offset_next += times.shape[0]
            dset_hit_time[hit_offset:hit_offset_next] = times
            dset_hit_pmt[hit_offset:hit_offset_next] = pmts
            hit_offset = hit_offset_next

        offset = offset_next
    f.close()
    print("saved", hit_offset, "hits in", offset, "good events (each with at least", min_hits, "hits)")
