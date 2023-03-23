
import cv2
import json
import numpy as np

from torch.utils.data import Dataset

class CircleDataset(Dataset):
    def __init__(self):
        self.data = []
        self.root = './ComposeNet/trainingdata/fill50k/'
        self.name = "CircleDataset"
        with open(self.root + 'prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(self.root + source_filename)
        target = cv2.imread(self.root + target_filename)

        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        source = source.astype(np.float32) / 255.0

        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)