import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset_pl import testDataset
from fer_pl import LightningModel

idx_to_class = {
    0: "angry",
    1: "anxiety",
    2: "happy",
    3: "hurt",
    4: "neutral",
    5: "sad",
    6: "surprise",
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
                    int(box[0][j]),
                    int(box[1][j]),
                    int(box[2][j]),
                    int(box[3][j]),
                ]

    return bbox_dict
