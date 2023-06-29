import pickle

def load_pkl(path):
    f = open(path,'rb')
    info = pickle.load(f)
    return info

train_path = '/data/yuqi_wang/nuscenes/occ3d-nus/occ_infos_temporal_train.pkl'
val_path = '/data/yuqi_wang/nuscenes/occ3d-nus/occ_infos_temporal_val.pkl'
train_val_path = '/data/yuqi_wang/nuscenes/occ3d-nus/occ_infos_temporal_trainval.pkl'

train_info = load_pkl(train_path)
val_info = load_pkl(val_path)

trainval_info = train_info
trainval_info['infos'].extend(val_info['infos'])

with open(train_val_path, 'wb') as f:
    pickle.dump(trainval_info, f)