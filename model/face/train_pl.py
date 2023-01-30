from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

# import mlflow.pytorch
# from mlflow import MlflowClient

from fer_pl import LightningModel


# def print_auto_logged_info(r):

#     tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
#     artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
#     print("run_id: {}".format(r.info.run_id))
#     print("artifacts: {}".format(artifacts))
#     print("params: {}".format(r.data.params))
#     print("metrics: {}".format(r.data.metrics))
#     print("tags: {}".format(tags))


def main():
    seed_everything(7)

    model = LightningModel()
    # AVAIL_GPUS = min(1, torch.cuda.device_count())

    trainer = Trainer(
        max_epochs=50,
        # val_check_interval = 1,
        accelerator="gpu",
        logger=CSVLogger(save_dir="./logs/"),
        callbacks=[
            ModelCheckpoint(
                filename="best_val_acc",
                verbose=True,
                save_last=True,
                save_top_k=1,
                monitor="val_acc",
                mode="max",
            ),
            LearningRateMonitor(logging_interval="step"),
            TQDMProgressBar(refresh_rate=10),
        ],
    )

    trainer.fit(model)

    # mlflow.pytorch.autolog()

    # with mlflow.start_run() as run:
    #     trainer.fit(model)

    # print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))


if __name__ == "__main__":
    main()
