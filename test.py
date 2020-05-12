import torch
import pandas as pd
from torch.utils.data import Dataset
import albumentations
import numpy as np
from PIL import Image
import os

MODEL = MODELS_DISPATCH[os.environ.get("MODEL")]

MODEL_MEAN = ast.literal_eval(os.environ.get("MODEL_MEAN"))
MODEL_STD = ast.literal_eval(os.environ.get("MODEL_STD"))

IMG_HEIGHT = int(os.environ.get("IMG_HEIGHT"))
IMG_WIDTH = int(os.environ.get("IMG_WIDTH"))

BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))


class catsvsdogsTest(Dataset):
    def __init__(self, img_height, img_width, mean, std):
        super(catsvsdogsTest, self).__init__()

        self.files = os.listdir('datasets/test/')
        self.aug = albumentations.Compose([
            albumentations.Resize(img_height, img_width, always_apply = True),
            albumentations.Normalize(mean, std, always_apply = True)
        ])
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, item):
        files = self.files[item]
        images = Image.open(os.path.join("datasets/test", self.files[item]))
        images =self.aug(image = np.array(images))['image']
        images = np.transpose(images, (2, 0, 1)).astype(np.float32)
        
        return {
            "images" : torch.tensor(images, dtype = torch.float),
            "files": files
        }

test_data = catsvsdogsTest(img_height = IMG_HEIGHT,
                            img_width = IMG_WIDTH,
                            mean= MODEL_MEAN,
                            std = MODEL_STD)
    
testloader = torch.utils.data.DataLoader(test_data, batch_size = BATCH_SIZE,
                                         shuffle  = False, num_workers = 0)




PATH = "model/resnet34_0.pth"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MODELS_DISPATCH[MODEL](pretrain = True)
model.to(DEVICE)


checkpoint = torch.load(PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])

result = []
files = []
model.eval()
for data in testloader:
    images, files = data["images"], data["files"]
    outputs = model(images.to(DEVICE))
    files.append(files[0])
    result.extend(outputs.tolist())

result_array = np.array(result)
result_array = np.clip(result_array, 0.01, 0.99) 
df = pd.DataFrame({"id":files_list, 'label':result_array})
df["id"] = df["id"].str.split(".").str[0]
df["id"] = df["id"].astype(int)
sort_df = df.sort_values(by=['id'], ignore_index= True)
sort_df.to_csv("submissions/submission__val_0.csv", index=False)