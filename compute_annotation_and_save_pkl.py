import os
import pandas as pd
import numpy as np
import pickle
from collections import Counter

def annotation(tmp_map, tmp_lactate, tmp_drug):
    
    # level 1 flag
    level_1_flag_CF = np.logical_and(np.logical_and(tmp_map > 65, tmp_drug == 0), tmp_lactate <= 2)
    level_1_flag_nonCF = np.logical_or(np.logical_and(tmp_lactate > 2, tmp_drug > 0), tmp_map <= 65)
    n_points = len(level_1_flag_CF)
    # level 2 flag
    width = 4
    level_2_flag_CF = [0]*width
    level_2_flag_nonCF = [0]*width
    for i in range(width, n_points-width):
        if np.sum(level_1_flag_CF[i-width:i+width]) >= 6:
            level_2_flag_CF.append(1)
        else:
            level_2_flag_CF.append(0)
        if np.sum(level_1_flag_nonCF[i-width:i+width]) >= 6:
            level_2_flag_nonCF.append(1)
        else:
            level_2_flag_nonCF.append(0)
        
    return np.array(level_2_flag_CF)[lag:], np.array(level_2_flag_nonCF)[lag:]
    
if __name__ == "__main__":
    lag = 8*12 # 8 hours, 5 mins each point
    csv_path = 'data/imputed_stage/csv'
    for fname in os.listdir(csv_path):
        tmp_df = pd.read_csv(os.path.join(csv_path, fname))
        tmp_pids = np.unique(tmp_df.patientid.values)
        for pid in tmp_pids:
            tmp_data = tmp_df[tmp_df.patientid==pid]
            # vars for Circulatory state annotation
            tmp_map = tmp_data.vm5.values
            tmp_lactate = (tmp_data.vm136.values + tmp_data.vm146.values)/2
            tmp_drug = np.array(np.sum(tmp_data.iloc[:,-5:-1].values, axis=1) > 0, dtype=float)
            flag_CF, flag_nonCF = annotation(tmp_map, tmp_lactate, tmp_drug)

            tmp_vm = tmp_data.iloc[:len(flag_CF),2:-5].values
            res = {'data':tmp_vm, 'flag_CF':flag_CF, 'flag_nonCF':flag_nonCF}
            with open('data/imputed_stage/pkl/{}.pkl'.format(pid), 'wb') as fout:
                pickle.dump(res, fout)

