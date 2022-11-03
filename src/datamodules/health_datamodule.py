from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class MyDataset(Dataset):

    def __init__(self, input_dataframe, split="train", target="Suicidio", ignore_columns=[], train_ratio=0.8):

        self.split = split
        self.target = target
        self.ignore_columns = ignore_columns

        for coll in self.ignore_columns:
            if coll in input_dataframe.columns:
                input_dataframe = input_dataframe.drop(coll, axis=1)

        self.classification_dim = len(input_dataframe[self.target].unique())
        self.data_dim = len(input_dataframe.columns) - 1
        self.embbeding_dim = input_dataframe.max().max() + 1

        y = input_dataframe[target].values
        x = input_dataframe.drop(target, axis = 1).values

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=1-train_ratio, random_state=42)

    def __len__(self):
        if self.split == "train":
            return len(self.x_train)
        elif self.split == "test":
            return len(self.x_test)
        else:
            raise ValueError("Split must be train or test")

    def __getitem__(self,idx):
        # target = torch.zeros(self.classification_dim)

        if self.split == "train":
            features = torch.tensor(self.x_train[idx])
            target = torch.tensor(self.y_train[idx])

        elif self.split == "test":
            features = torch.tensor(self.x_test[idx])
            target = torch.tensor(self.y_test[idx])

        else:
            raise ValueError("Split must be train or test")

        return features, target
        
class HEALTHDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        target: str = "Suicidio",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """

        dataframe = pd.read_csv('../data/final.csv', sep=';')
        selected = ['Suicidio', 'Trabalho e interesses', 'Apetite', 'Sentimentos_culpa', 'Perda de insights',
                    'Ansiedade somática', 'Ansiedade', 'Perda de peso', 'Lentidao pensamento e fala',
                    'Hipocondriase', 'Energia', 'Libido',  'sexo','Pontuação total', 'Deprimido']

        df_suic = dataframe[selected]

        df_suic['sexo'].replace({'M': 0, 'F': 1}, inplace=True)
        df_suic['sexo'].fillna(0, inplace=True)

        df_suic.dropna(inplace=True)
        df_suic = df_suic.astype(int)

        self.data_train = MyDataset(df_suic, split="train", target="Suicidio", ignore_columns=[], train_ratio=0.8)
        self.data_val = MyDataset(df_suic, split="test", target="Suicidio", ignore_columns=[], train_ratio=0.8)
        self.data_test = MyDataset(df_suic, split="test", target="Suicidio", ignore_columns=[], train_ratio=0.8)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "health_data.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
