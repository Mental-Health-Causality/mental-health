import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader 
from sklearn.model_selection import train_test_split

import importlib
if importlib.util.find_spec('ipywidgets') is not None:
    from tqdm.auto import tqdm
else:
    from tqdm import tqdm
    
dataframe = pd.read_csv('../data/final.csv', sep=';')
print(dataframe.head())

# all features
#selected = ['Suicidio','sexo', 'Estado_civil', 'Tipo_Resid','idade',
#                   'Alcoolatra', 'Droga', 'Suic_familia', 'Dep_familia',
#                    'Alc_familia', 'Drog_familia',
#                    'Neuro', 'psiquiatrica', 'Anos educacao formal', 'Capaz de desfrutar das coisas',
#                    'Impacto de sua familia e amigos',
#                    'Capaz de tomar decisões importantes', 'Estudante',
#                    'Insonia',
#                    'Deprimido', 'Ansiedade',
#                    'Perda de insights', 'Apetite', 'Perda de peso', 'Ansiedade somática',
#                   'Hipocondriase', 'Sentimentos_culpa', 
#                    'Trabalho e interesses', 'Energia', 'Lentidao pensamento e fala',
#                    'Agitação', 'Libido', 'TOC']

# causal graph: 8 features
selected = ['Suicidio', 'Drog_familia', 'Suic_familia',
                    'Capaz de tomar decisões importantes', 'Estudante',
                    'Hipocondriase', 'Sentimentos_culpa', 
                    'Trabalho e interesses', 'Energia']


dataframe['sexo'].replace({'M': 0, 'F': 1}, inplace=True)
dataframe['sexo'].fillna(0, inplace=True) # Talvez jogar fora

df_suic = dataframe[selected]

df_suic.dropna(inplace=True)
df_suic = df_suic.astype(int)


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
      target = torch.zeros(self.classification_dim) 
      if self.split == "train":
          target[self.y_train[idx]] = 1
          return (torch.tensor(self.x_train[idx]), target) 
      elif self.split == "test":
          target[self.y_test[idx]] = 1
          return (torch.tensor(self.x_test[idx]), target)
      else:
          raise ValueError("Split must be train or test")

# instanciando o dataset 
train_dataset = MyDataset(df_suic, split="train", target="Suicidio", ignore_columns=[], train_ratio=0.8)
test_dataset = MyDataset(df_suic, split="test", target="Suicidio", ignore_columns=[], train_ratio=0.8)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False) 
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

#######################################################################################################
## Define a MLP model with N layers: rede neural de 2 camadas

import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, n_layers=2):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
    
        self.layers = nn.ModuleList() 
        self.layers.append(nn.Linear(self.input_dim, self.hidden_dim)) 
        for i in range(self.n_layers - 1):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.layers.append(nn.Dropout(0.5)) 
            self.layers.append(nn.LeakyReLU())
            
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)           
        return x

###########################################################################################################
# Define a Model with a embbeding layer and a MLP

class ClassificationModel(nn.Module):
    def __init__(self, input_dim, output_dim, embbeding_dim, hidden_out, hidden_dim=128, n_layers=2):   
        super(ClassificationModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embbeding_dim = embbeding_dim
        self.embbeding_out = hidden_out
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embbeding_layer = nn.Embedding(self.embbeding_dim, self.embbeding_out) 
        self.mlp = MLP(self.input_dim * self.embbeding_out, self.output_dim, self.hidden_dim, self.n_layers) 
        
    def forward(self, x):
        x = self.embbeding_layer(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)
        ## classification
        x = F.softmax(x, dim=1)
        return x

###########################################################################################################
# test the model
example_batch = next(iter(train_loader))
example_data, example_targets = example_batch

model = ClassificationModel(train_dataset.data_dim, train_dataset.classification_dim, train_dataset.embbeding_dim, hidden_out=20, hidden_dim=128, n_layers=4) # Cadar tirou
print('model:', model, '\n')

print("Batch shape:", example_data.shape,'\n') 
res = model(example_data)
print("Output shape:", res.shape,'\n')

###########################################################################################################
## Make Lightning Module
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

class BaseModel(LightningModule):
    """A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Validation loop (validation_step)
        - Train loop (training_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)
    """

    def __init__(self, input_dim, output_dim, embedding_dim, embedding_out, hidden_dim):
        super().__init__()
        self.model = ClassificationModel(input_dim, output_dim, embedding_dim, embedding_out, hidden_dim=hidden_dim, n_layers=2)
        self.lr = 1e-3 

        self.save_hyperparameters()
        
        self.accuracy = Accuracy()
        
        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        # for averaging loss across batches
        #self.train_loss = MeanMetric()
        #self.val_loss = MeanMetric()
    
    def step(self, batch):
        x, y = batch
        y_hat = self.model(x).squeeze().float()
        # loss function
        loss = F.binary_cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y.int())
        return loss, acc
   
    def training_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log('train_loss', loss, prog_bar=True)        
        self.log('train_acc', acc,  prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log('val_loss', loss)      
        self.log('val_acc', acc,  prog_bar=True)
        return loss
    
    # gradiente
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

###########################################################################################################
# Import trainer
from pytorch_lightning.trainer import Trainer

# Initialize model
model = BaseModel(input_dim=train_dataset.data_dim, output_dim=train_dataset.classification_dim, embedding_dim=100, embedding_out=64, hidden_dim=128)
print('model:', model,'\n')

###########################################################################################################
# Import callbacks
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

print("Cuda:",torch.cuda.is_available())

# Initialize callbacks

# Salve o modelo periodicamente monitorando uma quantidade.
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints/', 
    filename='best-checkpoint', 
    save_top_k=1, 
    mode='min',
)

# Monitore uma métrica e interrompa o treinamento quando ela parar de melhorar.
early_stopping = EarlyStopping(
    monitor='val_loss', 
    min_delta=0.05, 
    patience=10, 
    verbose=False, 
    mode='min' 
)

callbacks = [checkpoint_callback, early_stopping]
# callbacks = []


# Initialize a trainer
trainer = Trainer(
    accelerator='gpu', 
    devices=1, 
    check_val_every_n_epoch=10, 
    log_every_n_steps=10, 
    callbacks=callbacks, 
    auto_lr_find=True, 
    enable_progress_bar=False
    ) 

# Train the model
trainer.fit(model, train_loader, test_loader)
