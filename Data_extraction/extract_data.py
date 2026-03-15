#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import pickle

sys.path.append('/home/Crystal_plasticity/Phenomeno/Poly/Cu/extract_data')
import DAMASK_helper


# ==============================
# PARAMETERS
# ==============================

num_runs   = 2000
batch_size = 1000
grid_dime  = [100,100]

result_path = '/home/Crystal_plasticity/Phenomeno/Poly/Cu/extract_data/'
data_path   = '/home/Crystal_plasticity/Phenomeno/Poly/Cu/dump_data/'

os.makedirs(data_path,exist_ok=True)

num_batches = num_runs // batch_size


# ==============================
# RESULT KEYS
# ==============================

res_filter = 'elem'

res_items = {
'elem':['elem'],
'position':['1_pos','2_pos','3_pos'],
'stress':['1_p','2_p','3_p','4_p','5_p','6_p','7_p','8_p','9_p'],
'defGrad':['1_f','2_f','3_f','4_f','5_f','6_f','7_f','8_f','9_f'],
'plasticdefGrad':['1_fp','2_fp','3_fp','4_fp','5_fp','6_fp','7_fp','8_fp','9_fp'],
'elasticdefGrad':['1_fe','2_fe','3_fe','4_fe','5_fe','6_fe','7_fe','8_fe','9_fe'],
'velGrad':['1_lp','2_lp','3_lp','4_lp','5_lp','6_lp','7_lp','8_lp','9_lp'],
'texture':['texture']
}

filter_ids='ALL'


tensor_sets = {
"p":["1_p","2_p","3_p","4_p","5_p","6_p","7_p","8_p","9_p"],
"f":["1_f","2_f","3_f","4_f","5_f","6_f","7_f","8_f","9_f"],
"fp":["1_fp","2_fp","3_fp","4_fp","5_fp","6_fp","7_fp","8_fp","9_fp"],
"fe":["1_fe","2_fe","3_fe","4_fe","5_fe","6_fe","7_fe","8_fe","9_fe"],
"lp":["1_lp","2_lp","3_lp","4_lp","5_lp","6_lp","7_lp","8_lp","9_lp"],
"texture":["texture"]
}


# ==============================
# MERGE SYMMETRIC COMPONENTS
# ==============================

def merge_components(keys,data_dict):

    result=[]

    pair_map={
    '2_p':'4_p','4_p':'2_p',
    '3_p':'7_p','7_p':'3_p',
    '6_p':'8_p','8_p':'6_p',
    '2_f':'4_f','4_f':'2_f',
    '3_f':'7_f','7_f':'3_f',
    '6_f':'8_f','8_f':'6_f'
    }

    for key in keys:

        if key in ['1_p','5_p','9_p','1_f','5_f','9_f']:

            tensor=np.array(data_dict[key]).reshape([-1]+grid_dime)-1

        elif key in pair_map:

            paired=pair_map[key]

            tensor=(np.array(data_dict[key]).reshape([-1]+grid_dime)+
                    np.array(data_dict[paired]).reshape([-1]+grid_dime))/2

        else:

            tensor=np.array(data_dict[key]).reshape([-1]+grid_dime)

        result.append(tensor)

    return np.stack(result,axis=-1).reshape([-1]+grid_dime+[3,3])


# ==============================
# BATCH LOOP
# ==============================

for b in range(num_batches):

    start=b*batch_size
    end=(b+1)*batch_size

    print("\n==============================")
    print("Processing batch",b+1,"/",num_batches)
    print("Runs:",start,"to",end-1)
    print("==============================")

    # ---------- READ RESULT FILES ----------

    result_file_template=result_path+'/d_INDEX_tension_inc100.txt'

    result_files={i:result_file_template.replace('INDEX',str(i))
                  for i in range(start,end)}

    R_=DAMASK_helper.read_DAMASK_results(result_files,res_filter,res_items,filter_ids)


    # ---------- READ MICROSTRUCTURE ----------

    geom_file_template=result_path+'/d_INDEX.geom'

    I_=[DAMASK_helper.readin_microstructure(
        geom_file_template.replace('INDEX',str(i)))[0][:,1:4]
        for i in range(start,end)]

    I_=np.array(I_)

    X=np.radians(I_.reshape([-1]+grid_dime+[3]))
    X=np.flip(X,axis=3)


    # ---------- BUILD TENSORS ----------

    p_tensor = merge_components(tensor_sets['p'],R_)
    f_tensor = merge_components(tensor_sets['f'],R_)

    fp_tensor=np.stack(
        [np.array(R_[k]).reshape(batch_size,*grid_dime)
        for k in tensor_sets['fp']],axis=-1
    ).reshape(batch_size,*grid_dime,3,3)

    fe_tensor=np.stack(
        [np.array(R_[k]).reshape(batch_size,*grid_dime)
        for k in tensor_sets['fe']],axis=-1
    ).reshape(batch_size,*grid_dime,3,3)

    lp_tensor=np.stack(
        [np.array(R_[k]).reshape(batch_size,*grid_dime)
        for k in tensor_sets['lp']],axis=-1
    ).reshape(batch_size,*grid_dime,3,3)

    texture_tensor=np.array(R_['texture']).reshape(batch_size,*grid_dime,1)


    # ---------- SAVE BATCH ----------

    np.save(data_path+f"p_batch_{b}.npy",p_tensor)
    np.save(data_path+f"f_batch_{b}.npy",f_tensor)
    np.save(data_path+f"fp_batch_{b}.npy",fp_tensor)
    np.save(data_path+f"fe_batch_{b}.npy",fe_tensor)
    np.save(data_path+f"lp_batch_{b}.npy",lp_tensor)
    np.save(data_path+f"texture_batch_{b}.npy",texture_tensor)
    np.save(data_path+f"X_batch_{b}.npy",X)

    print("Batch saved.")


# ==============================
# MERGE ALL BATCH FILES
# ==============================

print("\nMerging batches...")

def merge(prefix):

    arrays=[]

    for b in range(num_batches):

        file=data_path+f"{prefix}_batch_{b}.npy"
        arrays.append(np.load(file))

    merged=np.concatenate(arrays,axis=0)

    np.save(data_path+f"{prefix}_combined.npy",merged)

    print(prefix,"shape:",merged.shape)


merge("p")
merge("f")
merge("fp")
merge("fe")
merge("lp")
merge("texture")
merge("X")


print("\nAll batches merged successfully.")
