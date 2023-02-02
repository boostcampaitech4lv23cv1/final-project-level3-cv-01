import os
import sys
sys.path.append(os.getcwd())

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import pandas as pd

from model.face.fer_pl import LightningModel
from model.face.dataset_pl import testDataset

idx_to_class = {
    0: "angry",
    1: "anxiety",
    2: "happy",
    3: "hurt",
    4: "neutral",
    5: "sad",
    6: "surprise",
}

class_to_posneg = {
    0: "negative",
    1: "negative",
    2: "positive",
    3: "negative",
    4: "positive",
    5: "negative",
    6: "negative",
}


def inference(batch_size, model_ckpt_name, test_data_dir):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = LightningModel.load_from_checkpoint(model_ckpt_name)
    model = model.to(device)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    testdataset = testDataset(test_data_dir, transform)
    testloader = DataLoader(testdataset, batch_size=batch_size, shuffle=False)

    bbox_dict = {}

    model.eval()
    with torch.no_grad():
        for i in tqdm(testloader):
            input, box, path = i
            input = input.to(device)
            pred = model(input)
            pred = pred.argmax(dim=1).cpu().tolist()

            for j in range(len(box[0])):
                bbox_dict[path[j]] = [
                    pred[j],
                    idx_to_class[pred[j]],
                    class_to_posneg[pred[j]],
                    int(box[0][j]),
                    int(box[1][j]),
                    int(box[2][j]),
                    int(box[3][j]),
                ]

    df = pd.DataFrame(
        {
            "frame":bbox_dict.keys(),
            "emotion":[r[1] for r in bbox_dict.values()],
            "posneg":[r[2] for r in bbox_dict.values()],
            "x":[r[3] for r in bbox_dict.values()],
            "y":[r[4] for r in bbox_dict.values()],
            "w":[r[5]-r[3] for r in bbox_dict.values()],
            "h":[r[6]-r[4] for r in bbox_dict.values()],
        }
    )

    return bbox_dict, df

if __name__=="__main__":
    result = inference(
        batch_size=4, 
        model_ckpt_name="/opt/ml/input/final-project-level3-cv-01/model/face/models/best_val_posneg_acc.ckpt", 
        test_data_dir="/opt/ml/input/final-project-level3-cv-01/db/vis_mhchoi_images"
    )
    df = pd.DataFrame({"frame":result.keys(),"values":[r[1] for r in result.values()]})
    print(df)