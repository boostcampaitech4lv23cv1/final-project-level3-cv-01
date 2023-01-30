import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import torch
from torch import optim, cuda, nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder

from dataset import customDataset


def seed_everything(seed=2022):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = ArgumentParser()

    # Custom args
    parser.add_argument("--random_seed", type=int, default=2022)

    # Conventional args
    parser.add_argument("--train_dir", type=str, default="/opt/ml/data/train")
    parser.add_argument("--valid_dir", type=str, default="/opt/ml/data/valid")
    parser.add_argument("--saved_dir", type=str, default="trained_models")

    parser.add_argument("--device", default="cuda" if cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--image_size", type=int, default=112)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--valid_batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--save_interval", type=int, default=1)

    args = parser.parse_args()

    return args


def train(
    random_seed,
    image_size,
    train_dir,
    valid_dir,
    train_batch_size,
    valid_batch_size,
    num_workers,
    num_classes,
    device,
    lr,
    epochs,
    saved_dir,
):

    seed_everything(random_seed)

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    valid_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # train_dataset = customDataset(train_dir, train_transform)
    # valid_dataset = customDataset(valid_dir, valid_transform)
    train_dataset = ImageFolder(train_dir, train_transform)
    valid_dataset = ImageFolder(valid_dir, valid_transform)

    # train_loader = DataLoader(train_dataset, train_batch_size, True, num_workers=num_workers, pin_memory=True)
    # valid_loader = DataLoader(valid_dataset, valid_batch_size, False, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_dataset, train_batch_size, True)
    valid_loader = DataLoader(valid_dataset, valid_batch_size, False)

    # model = torch.load('models/affectnet_emotions/enet_b2_8_best.pt')
    model = torch.load("models/affectnet_emotions/enet_b2_8.pt")
    # model = torch.load('models/pretrained_faces/state_vggface2_enet2.pt')
    model.classifier = nn.Linear(1408, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        epoch_accuracy = 0
        model.train()
        for data, label in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            acc = (output.argmax(dim=1) == label).float().sum()
            epoch_accuracy += acc
            epoch_loss += loss
        epoch_accuracy /= len(train_dataset)
        epoch_loss /= len(train_dataset)

        with torch.no_grad():
            model.eval()
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in tqdm(valid_loader):
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().sum()
                epoch_val_accuracy += acc
                epoch_val_loss += val_loss

        epoch_val_accuracy /= len(valid_dataset)
        epoch_val_loss /= len(valid_dataset)
        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )
        scheduler.step()
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)

        torch.save(model, os.path.join(saved_dir, "latest.pt"))

        if best_acc < epoch_val_accuracy:
            best_acc = epoch_val_accuracy
            torch.save(model, os.path.join(saved_dir, "best.pt"))

            print(f"Best acc:{best_acc}")
            print(
                f"val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
            )
        else:
            print(f"No best model Best acc:{best_acc}")


def main():
    args = parse_args()
    random_seed = args.random_seed
    image_size = args.image_size
    train_dir = args.train_dir
    valid_dir = args.valid_dir
    train_batch_size = args.train_batch_size
    valid_batch_size = args.valid_batch_size
    num_workers = args.num_workers
    num_classes = args.num_classes
    device = args.device
    lr = args.learning_rate
    epochs = args.epochs
    saved_dir = args.saved_dir
    train(
        random_seed,
        image_size,
        train_dir,
        valid_dir,
        train_batch_size,
        valid_batch_size,
        num_workers,
        num_classes,
        device,
        lr,
        epochs,
        saved_dir,
    )


if __name__ == "__main__":
    main()
