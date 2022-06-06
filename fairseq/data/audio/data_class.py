import os
import torch
import torchaudio
from torch.utils.data import Dataset


HD100_PATH = "/DB/HD_100h/"


class HD100h(Dataset):
    def __init__(self):
        super().__init__()
        self.manifest_dir = os.path.join(HD100_PATH, "data_tmp/HD_100h")
        self.data_dir = os.path.join(HD100_PATH, "data")
        self.text = os.path.join(HD100_PATH, "data_tmp/HD_100h/text")
        self.dataset = []

        with open(self.text, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                line_list = line.split()
                wav_str = line_list[0]
                dir1, dir2, name1, name2, name3 = wav_str.split("_")
                dir = dir1 + "_" + dir2
                name = name1 + "_" + name2 + "_" + name3
                wav_path = os.path.join(self.data_dir, dir, name)
                label = " ".join(line_list[1:])
                wav, _ = torchaudio.load(wav_path)
                self.dataset.append((wav, label))

    def __getitem__(self, index):
        wav = self.dataset[index][0]
        label = self.dataset[index][1]

        return wav, label

    def __len__(self):
        return len(self.dataset)