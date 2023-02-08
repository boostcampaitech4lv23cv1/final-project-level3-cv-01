import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder
import pytorch_lightning as pl


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        model_path: str = "./model/face/models/affectnet_emotions",
        backbone: str = "enet_b2_7.pt",
        num_classes: int = 7,
        batch_size: int = 128,
        lr: float = 1e-3,
        epochs: int = 50,
        input_size: int = 224,
        data_dir: str = "/opt/ml/final-project-level3-cv-01/airflow/face_dataset_train_valid",
    ) -> None:

        super(LightningModel, self).__init__()
        self.model_path = model_path
        self.backbone = backbone
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.input_size = input_size
        self.data_dir = data_dir

        self.__build_model()
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.train_posneg_acc = Accuracy(task="binary")
        self.valid_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.valid_posneg_acc = Accuracy(task="binary")

        self.save_hyperparameters()

    def __build_model(self):
        # backbone = torch.load(self.model_backbone_path)
        backbone = torch.load(
            "/opt/ml/final-project-level3-cv-01/model/face/models/affectnet_emotions/enet_b0_8_va_mtl.pt"
        )
        _layers = list(backbone.children())

        self.feature_extractor = nn.Sequential(*_layers[:-1])
        self.fc_layer = nn.Linear(_layers[-1].in_features, self.num_classes)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc_layer(x)
        return x

    def loss_fn(self, output, target):
        return self.loss_func(output, target)

    def training_step(self, train_batch, batch_idx):
        input, target = train_batch
        output = self.forward(input)
        loss = self.loss_fn(output, target)

        train_output = output.argmax(dim=1)
        train_acc = self.train_acc(train_output, target)

        output_posneg = torch.where((train_output == 2) | (train_output == 4), 1, 0)
        target_posneg = torch.where((target == 2) | (target == 4), 1, 0)
        train_posneg_acc = self.train_posneg_acc(output_posneg, target_posneg)

        logs = {
            "train_loss": loss,
            "train_acc": train_acc,
            "train_posneg_acc": train_posneg_acc,
        }

        output = {
            "loss": loss,
            "acc": train_acc,
            "posneg_acc": train_posneg_acc,
            "progress_bar": logs,
            "log": logs,
        }

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc",
            train_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_posneg_acc",
            train_posneg_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # return {'loss':loss, 'acc':train_acc, 'log':logs}
        return output

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["acc"] for x in outputs]).mean()
        avg_posneg_acc = torch.stack([x["posneg_acc"] for x in outputs]).mean()
        self.log("avg_train_loss", avg_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("avg_train_acc", avg_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "avg_posneg_acc", avg_posneg_acc, on_epoch=True, prog_bar=True, logger=True
        )
        # return {'avg_train_loss':avg_loss, 'avg_train_acc':avg_acc}

    def validation_step(self, valid_batch, batch_idx):
        input, target = valid_batch
        output = self.forward(input)
        loss = self.loss_fn(output, target)

        valid_output = output.argmax(dim=1)
        valid_acc = self.valid_acc(valid_output, target)

        output_posneg = torch.where((valid_output == 2) | (valid_output == 4), 1, 0)
        target_posneg = torch.where((target == 2) | (target == 4), 1, 0)
        valid_posneg_acc = self.valid_posneg_acc(output_posneg, target_posneg)

        logs = {
            "valid_loss": loss,
            "valid_acc": valid_acc,
            "valid_posneg_acc": valid_posneg_acc,
        }

        output = {
            "valid_loss": loss,
            "valid_acc": valid_acc,
            "valid_posneg_acc": valid_posneg_acc,
            "progress_bar": logs,
            "log": logs,
        }
        self.log("val_loss", loss)
        self.log("val_acc", valid_acc)
        self.log("valid_posneg_acc", valid_posneg_acc)

        # return {'val_loss':loss, 'val_acc':valid_acc}
        return output

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["valid_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["valid_acc"] for x in outputs]).mean()
        avg_posneg_acc = torch.stack([x["valid_posneg_acc"] for x in outputs]).mean()
        self.log("avg_valid_loss", avg_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("avg_valid_acc", avg_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            "avg_valid_posneg_acc",
            avg_posneg_acc,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {
            "avg_val_loss": avg_loss,
            "avg_val_acc": avg_acc,
            "avg_val_posneg_acc": avg_posneg_acc,
        }

    def configure_optimizers(self):
        featrue_extractor_parameters = list(
            filter(lambda p: p.requires_grad, self.feature_extractor.parameters())
        )
        fc_layer_parameters = list(
            filter(lambda p: p.requires_grad, self.fc_layer.parameters())
        )
        parameters = featrue_extractor_parameters + fc_layer_parameters
        optimizer = torch.optim.Adam(parameters, self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )

        return [optimizer], [scheduler]

    def prepare_data(self):
        transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.train_dataset = ImageFolder(
            "/opt/ml/final-project-level3-cv-01/airflow/face_dataset_train_valid/train",
            transform,
        )
        self.valid_dataset = ImageFolder(
            "/opt/ml/final-project-level3-cv-01/airflow/face_dataset_train_valid/val",
            transform,
        )

        class_to_idx = self.train_dataset.class_to_idx
        self.idx_to_class = dict()
        for key, value in class_to_idx.items():
            self.idx_to_class[value] = key

        # idx_to_class = {
        #     0: "angry",
        #     1: "anxiety",
        #     2: "happy",
        #     3: "hurt",
        #     4: "neutral",
        #     5: "sad",
        #     6: "surprise",
        # }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, self.batch_size, shuffle=False, num_workers=4
        )
