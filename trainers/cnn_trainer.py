import torch
import pytorch_lightning as pl
#from torchmetrics.regression import F1Score, AUROC, Recall, ROC, Accuracy, Precision, Specificity 
from models.crnn import CRNN

class CNNTrainer(pl.LightningModule):
    def __init__(
        self,
        model_name,
        weights=None,
        learning_rate=0.001,
        weight_decay=1e-4,
        gamma=2,
        num_classes = 223,
        num_channels = 1,
    ):
        """Initialize with pretrained weights instead of None if a pretrained model is required."""
        super().__init__()
        self.training_step_losses = []
        self.training_step_accuracy = []

        self.validation_step_losses = []
        self.validation_step_accuracy = []

        self.test_step_losses = []

        self.weights = weights
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gamma = gamma

        if num_classes == 2:
            self.num_classes = num_classes - 1
        elif num_classes > 2:
            self.num_classes = num_classes

        self.num_channels = num_channels

        self.model_name = model_name

        self.conv_reshape = []
        # models ###################
        if self.model_name == "crnn":
            self.model = self._get_model()


        # metrics ###################
        # self.f1 = F1Score(task=self.classification, num_classes=self.num_classes)
        # self.auroc = AUROC(task=self.classification, num_classes=self.num_classes)
        # self.recall = Recall(task=self.classification, num_classes=self.num_classes)
        # self.accuracy = Accuracy(task=, num_classes=self.num_classes)
        # self.precision = Precision(task=self.classification, num_classes=self.num_classes)
        # self.specifity = Specificity(task=self.classification, num_classes=self.num_classes)
        self.save_hyperparameters()

    def loss_fn(self, output, target):
        # 🪟 Clip the target values to be within the range [0, 1]
        #clipped_target = torch.clip(target, min=0, max=1)
        # 📉 Calculate the mean squared error loss
        
        #mses = torch.nn.functional.l1_loss(output, clipped_target, reduction='mean')
        mses = torch.nn.functional.l1_loss(output, target, reduction='mean')

        return mses

        
    def _get_model(self):
        model = CRNN(img_channel=1, img_width=512, img_height=512)

        return model


    # def _metrics(self, y_pred, y_true):
    #     f1 = self.f1(y_pred, y_true)
    #     auroc = self.auroc(y_pred, y_true)
    #     recall = self.recall(y_pred, y_true)
    #     precision = self.precision(y_pred, y_true)
    #     accuracy = self.accuracy(y_pred, y_true)
    #     specifity = self.specifity(y_pred, y_true)
    #     return accuracy

    def forward(self, imgs):
        output = self.model(imgs)
        #print(output)
        return output
    
    
    def configure_optimizers(self):
       
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, y_true, mask = batch
        

        y_pred = self.forward(inputs)
        y_pred = y_pred * mask.unsqueeze(-1)
        #print(mask.unique())
        #print(y_pred.unique())       
        # loss
    
        y_true = y_true.unsqueeze(-1)
        y_true = y_true * mask.unsqueeze(-1) 
        loss = self.loss_fn(y_pred, y_true)

        # accuracy = self._metrics(y_pred, y_true)

        self.training_step_losses.append(loss)
        # self.training_step_accuracy.append(accuracy)

        self.log("train_loss", loss, prog_bar=True, on_step=True)
        # self.log("train_accuracy", accuracy, prog_bar=True, on_step=True)
        # self.log("train_auroc", auroc, prog_bar=False, on_step=True)
        # self.log("train_precision", precision, prog_bar=False, on_step=True)
        # self.log("train_recall", recall, prog_bar=False, on_step=True)
        # self.log("train_f1", f1, prog_bar=False, on_step=True)
        # self.log("train_specifity", specifity, prog_bar=False, on_step=True)


        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.training_step_losses).mean()
        # avg_train_acc = torch.stack(self.training_step_accuracy).mean()

        self.log(
            "train_loss_epoch_end",
            avg_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        # self.log(
        #     "train_acc_epoch_end",
        #     avg_train_acc,
        #     prog_bar=True,
        #     on_step=False,
        #     on_epoch=True,
        # )
        return avg_loss

    def validation_step(self, batch, batch_idx):
        inputs, y_true, mask = batch

        y_pred = self.forward(inputs)
        y_pred = y_pred * mask.unsqueeze(-1)
     
    
        # loss
        y_true = y_true.unsqueeze(-1)
        y_true = y_true * mask.unsqueeze(-1) 
        loss = self.loss_fn(y_pred, y_true)

        # accuracy = self._metrics(y_pred, y_true)

        self.validation_step_losses.append(loss)
        # self.validation_step_accuracy.append(accuracy)

        self.log("val_loss", loss, prog_bar=True, on_step=True)
        # self.log("val_accuracy", accuracy, prog_bar=True, on_step=True)
        # self.log("val_auroc", auroc, prog_bar=False, on_step=True)
        # self.log("val_precision", precision, prog_bar=False, on_step=True)
        # self.log("val_recall", recall, prog_bar=False, on_step=True)
        # self.log("val_f1", f1, prog_bar=False, on_step=True)
        # self.log("val_specifity", specifity, prog_bar=False, on_step=True)


        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_losses).mean()
        # avg_train_acc = torch.stack(self.validation_step_accuracy).mean()

        self.log(
            "validation_loss_epoch_end",
            avg_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        # self.log(
        #     "validation_acc_epoch_end",
        #     avg_train_acc,
        #     prog_bar=True,
        #     on_step=False,
        #     on_epoch=True,
        # )
        return avg_loss

    # This is for binaryCE loss cuz one dimension 

    def predict_step(self, batch, batch_idx):
        inputs, mask = batch
        y_pred = self.forward(inputs)
        y_pred = y_pred * mask.unsqueeze(-1)
        # if self.classification == 'binary':
        #     y_pred = torch.nn.functional.sigmoid(y_pred)
        return y_pred