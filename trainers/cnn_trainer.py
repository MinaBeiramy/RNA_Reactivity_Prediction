import torch
import pytorch_lightning as pl
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from models.crnn import CRNN

class CNNTrainer(pl.LightningModule):
    def __init__(
        self,
        model_name,
        weights=None,
        learning_rate=0.001,
        weight_decay=1e-4,
        gamma=2,
        num_channels = 1,
    ):
        """Initialize with pretrained weights instead of None if a pretrained model is required."""
        super().__init__()
        self.training_step_losses = []
        self.training_step_mae = []
        self.training_step_rmse = []

        self.validation_step_losses = []
        self.validation_step_mae = []
        self.validation_step_rmse = []

        self.test_step_losses = []

        self.weights = weights
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gamma = gamma

        self.num_channels = num_channels

        self.model_name = model_name

        self.conv_reshape = []
        # models ###################
        if self.model_name == "crnn":
            self.model = self._get_model()


        # metrics ###################
        self.mae = MeanAbsoluteError()
        self.mse = MeanSquaredError()
        self.save_hyperparameters()

    def loss_fn(self, output, target):
        # 🪟 Clip the target values to be within the range [0, 1]
        clipped_target = torch.clip(target, min=0, max=1)
        # 📉 Calculate the mean squared error loss
        mses = torch.nn.functional.mse_loss(output, clipped_target, reduction='mean')
        return mses

        
    def _get_model(self):
        model = CRNN(img_channel=1, img_width=512, img_height=512)
        return model


    def _metrics(self, y_pred, y_true):
        clipped_target = torch.clip(y_true, min=0, max=1)
        mae = self.mae(y_pred, clipped_target)
        rmse = torch.sqrt(self.mse(y_pred, clipped_target))
        return mae, rmse

    def forward(self, imgs):
        output = self.model(imgs)
        return output.contiguous()
    
    
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
        # loss
    
        y_true = y_true.unsqueeze(-1)
        y_true = y_true * mask.unsqueeze(-1) 

        loss = self.loss_fn(y_pred, y_true)
        mae, rmse = self._metrics(y_pred, y_true)

        self.training_step_losses.append(loss.detach().cpu())
        self.training_step_mae.append(mae.detach().cpu())
        self.training_step_rmse.append(rmse.detach().cpu())

        self.log("train_loss", loss, prog_bar=True, on_step=True)
        self.log("train_mae", mae, prog_bar=True, on_step=True)
        self.log("train_rmse", rmse, prog_bar=True, on_step=True)

        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.training_step_losses).mean()
        avg_mae = torch.stack(self.training_step_mae).mean()
        avg_rmse = torch.stack(self.training_step_rmse).mean()

        self.log(
            "train_loss_epoch_end",
            avg_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_mae_epoch_end",
            avg_mae,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_rmse_epoch_end",
            avg_rmse,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return avg_loss

    def validation_step(self, batch, batch_idx):
        inputs, y_true, mask = batch
        y_pred = self.forward(inputs)
        y_pred = y_pred * mask.unsqueeze(-1)
    
        # loss
        y_true = y_true.unsqueeze(-1)
        y_true = y_true * mask.unsqueeze(-1) 
        #print (y_pred.shape, y_true.shape)

        loss = self.loss_fn(y_pred, y_true)
        mae, rmse = self._metrics(y_pred, y_true)

        self.validation_step_losses.append(loss.detach().cpu())
        self.validation_step_mae.append(mae.detach().cpu())
        self.validation_step_rmse.append(rmse.detach().cpu())
        self.log("val_loss", loss, prog_bar=True, on_step=True)
        self.log("val_mae", mae, prog_bar=True, on_step=True)
        self.log("val_rmse", rmse, prog_bar=True, on_step=True)
        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_losses).mean()
        avg_mae = torch.stack(self.validation_step_mae).mean()
        avg_rmse = torch.stack(self.validation_step_rmse).mean()

        self.log(
            "validation_loss_epoch_end",
            avg_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "validation_mae_epoch_end",
            avg_mae,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "validation_rmse_epoch_end",
            avg_rmse,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return avg_loss
    
    def predict_step(self, batch, batch_idx):
        inputs, mask, seq_len = batch
        y_pred = self.forward(inputs)
        y_pred = y_pred.squeeze()
        # y_pred = y_pred * mask.reshape()
        # pads = 512 - seq_len 
        # y_pred = y_pred[int(pads/2):int(pads/2) + seq_len]

        return y_pred, seq_len
