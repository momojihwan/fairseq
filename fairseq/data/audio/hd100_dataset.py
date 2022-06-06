import os
from typing import Tuple, Union
from pathlib import Path
import torch
import torchaudio
from torch.utils.data import Dataset

def load_hd100_item(
    wav_id, path, manifest_path
):
    with open(manifest_path + "wav.scp", "r", encoding="utf-8") as f:
        for line in f:
            utter_id = line.strip().split("/")[-1]
            utterance_id = utter_id.strip().split(".")[0]
            
            if wav_id == utterance_id:              
                fileid, _ = line.strip().split("\t")
                break
    spk_id = '_'.join(fileid.strip().split("_")[:2])

    file_audio = os.path.join(path, spk_id, wav_id + '.wav')
    file_text = os.path.join(path, spk_id, utterance_id + ".txt")
    
    waveform, sample_rate = torchaudio.load(file_audio)
    with open(file_text, "r", encoding="utf-8") as ft:
        for line in ft:
            utterance = line.strip()
        
    return (
        waveform,
        sample_rate,
        utterance,
        spk_id,
        utterance_id
    )

def load_hd500_item(
    wav_id, path, manifest_path
):
    with open(manifest_path + "wav.scp", "r", encoding="utf-8") as f:
        for line in f:
            utter_id = line.strip().split("/")[-1]
            utterance_id = utter_id.strip().split(".")[0]
            
            if wav_id == utterance_id:              
                fileid, _ = line.strip().split("\t")
                break
    spk_id = '_'.join(fileid.strip().split("_")[:3])

    file_audio = os.path.join(path, spk_id, wav_id + '.wav')
    file_text = os.path.join(path, spk_id, utterance_id + ".trn")
    
    waveform, sample_rate = torchaudio.load(file_audio)
    with open(file_text, "r", encoding="utf-8") as ft:
        for line in ft:
            utterance = line.strip()
        
    return (
        waveform,
        sample_rate,
        utterance,
        spk_id,
        utterance_id
    )

class HD100h(Dataset):
    def __init__(self, root: Union[str, Path], url):
        super().__init__()
        self.manifest_path = os.path.join(root, "data_tmp/HD_100h/")
        self._path = os.path.join(root, "HD100h", url)
        self._walker = sorted(str(p.stem) for p in Path(self._path).glob('*/*' + ".wav"))

    def __getitem__(self, index):
        fileid = self._walker[index]
        return load_hd100_item(fileid, self._path, self.manifest_path)

    def __len__(self):
        return len(self._walker)

class HD500h(Dataset):
    def __init__(self, root, url):
        super().__init__()
        self.manifest_path = os.path.join(root, "HD_500h/")
        self._path = os.path.join(root, "data")
        self._walker = sorted(str(p.stem) for p in Path(self._path).glob('*/*' + ".wav"))

    def __getitem__(self, index):
        fileid = self._walker[index]
        return load_hd500_item(fileid, self._path, self.manifest_path)

    def __len__(self):
        return len(self._walker)