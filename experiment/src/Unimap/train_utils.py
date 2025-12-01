
import pickle
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

def load_pkl_with_labels(data_dir, arch, csv_path, cache_file):
    cache_path = Path(cache_file) / f"{arch}_cached.pkl"
    
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    pkl_dir = Path(data_dir) / arch
    df = pd.read_csv(csv_path)
    df_arch = df[df['CPU'] == arch]
    
    label_encoder = LabelEncoder()
    df_arch['encoded_label'] = label_encoder.fit_transform(df_arch['label'])
    file_to_label = dict(zip(df_arch['file_name'], df_arch['encoded_label']))
    
    data_with_labels = []
    for pkl_file in sorted(pkl_dir.glob("*.pkl")):
        file_name = pkl_file.stem
        if file_name in file_to_label:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                label = file_to_label[file_name]
                data_with_labels.append((data, label))
    
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(data_with_labels, f)
    
    return data_with_labels
