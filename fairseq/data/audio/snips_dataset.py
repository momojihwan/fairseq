import os
import json
import torchaudio
from torch.utils.data import Dataset

class HEYSNIPS(Dataset):
    def __init__(self, root, subset: str = None):
        super().__init__()
        self.wav_dir = os.path.join(root, subset)
        self.json_name = subset + ".json"
        self.json_path = os.path.join(root, subset, self.json_name)
        self.dataset = []
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for entry in data:
                wav_path = entry["audio_file_path"]
                wv, sr = torchaudio.load(wav_path)
                tu = (wv, sr, entry["is_hotword"], entry["worker_id"], entry["id"], wav_path)
                self.dataset.append(tu)

    def __getitem__(self, index):
        wav = self.dataset[index][0]
        sample_rate = self.dataset[index][1]
        label = self.dataset[index][2]
        spk_id = self.dataset[index][3]
        utt_id = self.dataset[index][4]
        wav_path = self.dataset[index][5]

        return wav, sample_rate, label, spk_id, utt_id, wav_path

    def __len__(self):
        return len(self.dataset)