import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Batch


def collater(data_list):
    batch = {}
    name = [x["name"] for x in data_list]
    protac_graphs = [x["protac_graphs"] for x in data_list]
    ligase_pocket = [x["ligase_pocket"] for x in data_list]
    target_pocket = [x["target_pocket"] for x in data_list]
    feature = [x['feature'] for x in data_list]
    label = [x["label"] for x in data_list]
    sample_idx = [x["sample_idx"] for x in data_list]

    batch["name"] = name
    batch["protac_graphs"] = Batch.from_data_list(protac_graphs)
    batch["ligase_pocket"] = Batch.from_data_list(ligase_pocket)
    batch["target_pocket"] = Batch.from_data_list(target_pocket)
    batch["feature"] = torch.as_tensor(np.asarray(feature, dtype=np.float32))
    batch["label"]=torch.tensor(label)
    batch["sample_idx"] = torch.as_tensor(sample_idx, dtype=torch.long)
    return batch


class PROTACSet(Dataset):
    def __init__(self, name, protac_graphs, ligase_pocket, target_pocket, feature, label):
        super().__init__()
        n_common = min(len(name), len(protac_graphs), len(ligase_pocket), len(target_pocket), len(feature), len(label))
        self.indices = list(range(n_common))
        self.name = list(name)[:n_common]
        self.protac_graphs = protac_graphs
        self.ligase_pocket = ligase_pocket
        self.target_pocket = target_pocket
        self.feature = feature[:n_common]
        self.label = label[:n_common]
        self.n_common = n_common


    def __len__(self):
        return self.n_common

    def __getitem__(self, idx):
        i = self.indices[idx]
        sample = {
            "name": self.name[i],
            'protac_graphs': self.protac_graphs.get(i) if hasattr(self.protac_graphs, 'get') else self.protac_graphs[i],
            "ligase_pocket": self.ligase_pocket.get(i) if hasattr(self.ligase_pocket, 'get') else self.ligase_pocket[i],
            "target_pocket": self.target_pocket.get(i) if hasattr(self.target_pocket, 'get') else self.target_pocket[i],
            'feature': self.feature[i],
            "label": self.label[i],
            "sample_idx": i,
        }
        return sample
