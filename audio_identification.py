import librosa
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

from scipy.spatial import distance
from fingerprint import fingerprintBuilder
from tqdm import tqdm


class audioIdentification(fingerprintBuilder):
    """
    Audio identifier as defined in Philips algorithm.

    Inherits fingerprintBuilder since it requires its class functions.
    """
    def __init__(self,  print_logs=False):
        fingerprintBuilder.__init__(self, print_logs=False)
        self.print_logs = print_logs
        self.search_type = 'simple'
        self.hash_table = {}
        print(f"Initialized audio identification")
        
    def __call__(self, query_path, fingerprint_path, output_path):
        print(f"Reading fingerprints from {fingerprint_path}/fingerprints_{self.overlap_factor}.pkl")
        with open(f'{fingerprint_path}/fingerprints_{self.overlap_factor}.pkl', 'rb') as handle:
            self.song_db = pickle.load(handle)

        print(f"Loading fingerprints to hash table")
        for song_idx, fingerprint in self.song_db.items():
            self.hash_table = self.load_song_to_database(song_idx, fingerprint, self.hash_table)

        print(f"Matching queries in {query_path}...")
        output_file = open(f"{self.search_type}_{self.overlap_factor}_{output_path}","a")

        for query_file in tqdm(os.listdir(query_path)):
            if self.search_type == 'complex':
                matches = self.get_query_matches_complex(query_path, query_file)
            else:
                matches = self.get_query_matches_simple(query_path, query_file)
            sorted_matches = sorted(matches, key=matches.get, reverse=False)
            output_str = ','.join(sorted_matches[:3]).replace(',', '\t')
            output_file.write(f"{query_file}\t{output_str}\n")
        
        output_file.close()

    
        
    # Store fingerprints in hash table (python dictionary)
    def hash_subfingerprint(self, sub_fp):
        """
        Converts a 32-bit sub-fingerprint to a hex representation for easy lookup.
    
        E.g. np.array([1, 0, 0 ,0, 0, 1, 1, 1, 1, 0,
                        0, 0, 0, 0, 1, 1, 0, 0 ,0, 0,
                        1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 
                        0, 0]) 
            is converted to '0x87830f04'
        """
        return hex(int("".join(sub_fp.astype(str)), 2))
        
    def load_song_to_database(self, song_idx, fingerprint, lookup_dict={}):
        """
        Loads fingerprints of a single song in a hash table.
        
        Format:
        {<hashed key>: {
            <song_id1>: [frame_idx1, frame_idx2],
            <song_id2>: [frame_idx3],
            }
        Example:
        {
            "0x4be0de43": {
                'classical.00000.wav': [0],
                },
        }
        """
        for frame_idx in range(fingerprint.shape[0]):
            key = self.hash_subfingerprint((fingerprint[frame_idx]>0).astype(int))
            if key not in lookup_dict:
                lookup_dict[key] = {song_idx: [frame_idx]}
            elif song_idx in lookup_dict[key]:
                lookup_dict[key][song_idx].append(frame_idx)
            else:
                lookup_dict[key][song_idx] = [frame_idx]
        return lookup_dict
    
    
    
    def get_ber(self, query_fp, db_fp):
        """
        Calculate Bit Error Rate between 2 fingerprints.
        
        """
        total_dist = 0
        for sub_fp, db_fp in zip(query_fp, db_fp):
            total_dist += (distance.hamming(sub_fp, db_fp) * 32)  # Multiply by 32 since scipy package divides bit count by 32
        ber = total_dist/(256*32)
    
        return ber
    
    
    def get_query_matches_simple(self, query_path, song_idx):
        """
        Philips Search algorithm (Assumption: at least 1 sub-fingerprint with exact match)
        1. Get finger prints for each query file
        2. Loop through sub-fingerprints
        3. Match sub-fingerprint against db (exact match)
        4. Loop through matches and calculate minimum BER of each block
        
        """
        matches = {}

        
        melspec = self.get_melspec(f"{query_path}/{song_idx}")
        query_fp =  self.get_fingerprint(melspec)
        query_matches = []
        for sub_fp_idx in range(256, len(query_fp)):
            sub_fp = query_fp[sub_fp_idx]
            key = self.hash_subfingerprint(sub_fp.astype(int))
            if key in self.hash_table:
                # print(f"Found key {key} at index {sub_fp_idx} in hash table with exact match")
                # Loop through all songs with that sub-fingerprint
                for file_id, frame_ids in self.hash_table[key].items():
                    if matches.get(file_id) is None:
                        matches[file_id] = 1
                    for frame_id in frame_ids:
                        # Check if BER meets threshold requirements
                        fp_block = self.song_db[file_id][frame_id-256: frame_id]
                        query_fp_blk = query_fp[sub_fp_idx-256:sub_fp_idx]
                        ber = self.get_ber(query_fp_blk, fp_block)
                        
                        # Keep minimum BER only
                        matches[file_id] = min(matches[file_id], ber)
    
        return matches
    
    def is_matched(self, matches, actual, k=3):
        """
        Returns whether actual piece is found in matches as all levels of top-k
    
        E.g. if k=3 and piece if found at rank 2, then return [False, True, True].
        """
        match_at_k = []
        sorted_matches = sorted(matches, key=matches.get, reverse=False)
        # matched = False
        for i in range(k):
            if f"{actual.split('-')[0]}.wav" in sorted_matches[:i]:
                match_at_k.append(True)
            else:
                match_at_k.append(False)
        
        return match_at_k
    
    
    
    def get_query_matches_complex(self, query_path, query_file):
        """
        Philips Search algorithm (Swap unstable bits)
        1. Get finger prints for each query file
        2. Loop through sub-fingerprints
        3. Match sub-fingerprint against db (exact match)
        4. Loop through matches and calculate minimum BER of each block
        5. Loop through sub-fingerprints, but swapping out the k most unstable bits. Repeat 4 and 5.
        
        """
        matches = {}
        k=3
        melspec = self.get_melspec(f"{query_path}/{query_file}")
        query_fp =  self.get_fingerprint(melspec, encode=False) # Set encode=False to get raw energy diff values
        query_matches = []
    
        for sub_fp_idx in range(256, len(query_fp)):
            sub_fp = query_fp[sub_fp_idx]
            
            # Encode fingerprint values
            bin_fp = sub_fp > 0
    
            key = self.hash_subfingerprint(bin_fp.astype(int))
            least_reliable = np.argsort(np.abs(sub_fp))
    
            if key in self.hash_table:
                query_matches.append(key)
                # print(f"Found key {key} in hash table with exact match")
                
                for file_id, frame_ids in self.hash_table[key].items():
                    if matches.get(file_id) is None:
                        matches[file_id] = 1
                    for frame_id in frame_ids:
                        # Check if BER meets threshold requirements
                        fp_block = self.song_db[file_id][frame_id-256: frame_id]
                        query_fp_blk = (query_fp > 0).astype(int)[sub_fp_idx-256:sub_fp_idx]
                        ber = self.get_ber(query_fp_blk, fp_block)
                        
                        # Keep minimum BER only
                        matches[file_id] = min(matches[file_id], ber)
    
            # Check for least-reliable bit matches
            sec_bin_fp = bin_fp.copy()
            i = 0
    
            for i in range(k):
    
                least_conf_idx = least_reliable[i]
                sec_bin_fp[least_conf_idx] = ~sec_bin_fp[least_conf_idx]
                key = self.hash_subfingerprint(sec_bin_fp.astype(int))
                i += 1
    
                if key in self.hash_table:
                    query_matches.append(key)
                    # print(f'Found key {key} in hash table after switching {i} bits')
                    for file_id, frame_ids in self.hash_table[key].items():
                        if matches.get(file_id) is None:
                            matches[file_id] = 1
                            
                        for frame_id in frame_ids:
                            # Check if BER meets threshold requirements
                            fp_block = self.song_db[file_id][frame_id-256: frame_id]
                            query_fp_blk = (query_fp > 0).astype(int)[sub_fp_idx-256:sub_fp_idx]
                            ber = self.get_ber(query_fp_blk, fp_block)
                            
                            # Keep minimum BER only
                            matches[file_id] = min(matches[file_id], ber)
        return matches
    