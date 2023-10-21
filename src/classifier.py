from dataclasses import dataclass

from typing import List
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from sklearn.preprocessing import LabelBinarizer
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


class FTVectorizer:
    """
    Classe pour vectoriser les textes et les labels
    - Les vecteurs de représentation des textes constitueront l'entrée du modèle neuronal
    - Les vecteurs de représentation des labels sont utilisés pour calculer la valeur du loss lors
    de l'entraînement du modèle
    Dans cet exemple, on définit cette classe FTVectorizer pour une représentation de type sac-de-mots
    (Bag-of-Words ou BoW). Pour utiliser une vectrisation de type modèle de langage pré-entraîné (par
    exemple avec le modèle transformers 'cammenbert') il faut modifier le code ci-dessous.

    """

    def __init__(self):
        # Initialize tokenizer & label binarizer
        self.vectorizer = AutoTokenizer.from_pretrained('Yanzhu/bertweetfr-base')
        self.label_binarizer = LabelBinarizer()

    #############################################################################################
    # Ne pas modifier la signature (nom, arguments et type retourné) des méthodes suivantes
    # (mais vous pouvez modifier leur corps)
    #############################################################################################

    def fit(self, train_texts: List[str], train_labels: List[str]):
        self.label_binarizer.fit(train_labels)

    def input_size(self) -> int:
        """
        :return: The size of input vector representations
        """
        return self.vectorizer.vocab_size

    def output_size(self) -> int:
        """
        :return: The size of the output (label) vector representations
        """
        return len(self.label_binarizer.classes_)

    def vectorize_input(self, texts) -> List[torch.Tensor]:
        """
        Produces the vectorized representations of the input: these vectors will be the inputs to the
        neural network model
        :param texts:
        :return:
        """
        # Converts texts into tokens
        vects = [torch.tensor(self.vectorizer.encode(text, add_special_tokens=True, max_length=512, truncation=True))
                 for text in texts]
        return vects

    def vectorize_labels(self, labels) -> List[torch.Tensor]:
        """
        Produces the vectorized representations of the labels: these vectors will be the inputs to the
        the loss function used when training the neural network model
        :param labels:
        :return:
        """
        vects = self.label_binarizer.transform(labels)
        return [torch.from_numpy(vect).float() for vect in vects]

    def devectorize_labels(self, prediction_vects):
        return self.label_binarizer.inverse_transform(prediction_vects)

    def batch_collate_fn(self, batch_list):
        """
        :param batch_list:
        :return: the batch built from the list of examples batch_list
        """
        # batch_list is a list of tuples, each returned by the __get_item__() function of
        # the ReviewDataset class
        # create 2 separate lists for each element type in the tuples
        input_vects, label_vects = tuple(zip(*batch_list))
        # the batch will be a dictionary of tensors: a tensor for the input vectors, and another for the label_ids if
        # any
        batch = dict({})
        batch['input_vects'] = torch.nn.utils.rnn.pad_sequence(input_vects, batch_first=True, padding_value=1)
        if label_vects[0] is not None:
            batch['label_vects'] = torch.stack(label_vects).float()
        # return the batch as a dictionary of tensors
        return batch


@dataclass
class HyperParameters:
    batch_size: int = 10
    learning_rate: float = 1e-5
    max_epochs: int = 20
    dropout: float = 0.2
    weight_decay: float = 1e-6
    ad_eps: float = 1e-5
    # early stopping
    es_monitor: str = 'val_loss'
    es_mode: str = 'min'
    es_patience: int = 5
    es_min_delta: float = 0.1
    # checkpoint save and selection
    ckpt_monitor: str = 'val_loss'
    ckpt_mode: str = 'min'


HP = HyperParameters()


class FTClassifier(pl.LightningModule):

    #############################################################################################
    # Ne pas modifier la signature (nom, arguments et type retourné) des méthodes suivantes
    # Pour améliorer le modèle, modifiez seulement le contenu des méthodes (vous pouvez
    # rajouter de nouvelles méthodes si nécessaire)
    #############################################################################################

    def __init__(self, vectorizer: FTVectorizer):
        super().__init__()
        self.vectorizer = vectorizer
        # HuggingFace model to download from the hub
        hf_plm_name = 'Yanzhu/bertweetfr-base'
        # Problem type
        pb_type = 'multi_label_classification'
        # Number of labels
        num_labels = 3
        # Model configuration : dropout, model type, ...
        config = AutoConfig.from_pretrained(hf_plm_name, hidden_dropout_prob=HP.dropout, num_labels=num_labels,
                                            problem_type=pb_type)
        # Model
        self.lm = AutoModelForSequenceClassification.from_pretrained(hf_plm_name, config=config)
        # Loss function
        self.loss_fn = torch.nn.CrossEntropyLoss()
        # grouped parameters for the optimizer
        self.optimizer_grouped_parameters = None

    def forward(self, batch):
        out = self.lm(batch['input_vects'])
        return out.logits

    def training_step(self, batch, batch_idx):
        # training_step is called in PyTorch Lightning train loop
        y_hat = self.forward(batch)
        loss = self.loss_fn(y_hat, batch['label_vects'])
        return loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        self.optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.lm.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": HP.weight_decay,
            },
            {
                "params": [p for n, p in self.lm.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            }
        ]
        optimizer = torch.optim.AdamW(self.optimizer_grouped_parameters, lr=HP.learning_rate, eps=HP.ad_eps)
        return optimizer

    def validation_step(self, batch, batch_ix):
        # validation_step is called in PyTorch Lightning train loop
        y_hat = self.forward(batch)
        loss = self.loss_fn(y_hat, batch['label_vects'])
        self.log_dict({'val_loss': loss.item()}, on_step=False, on_epoch=True, reduce_fx='mean',
                      prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        y_hat = self.forward(batch)
        y_hat = F.softmax(y_hat, dim=1)
        return y_hat
