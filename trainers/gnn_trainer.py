<<<<<<< HEAD
import torch
import pytorch_lightning as pl
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from torch_geometric.nn.models import EdgeCNN

class GNNTrainer(pl.LightningModule):
    def __init__(
        self,
        model_name,
        num_channels,
        num_features,
        batch_size,
        edge_distance=4,
        weights=None,
        learning_rate=0.001,
        weight_decay=1e-4,
        gamma=2,
        
    ):
        """Initialize with pretrained weights instead of None if a pretrained model is required."""
        super().__init__()
        self.training_step_losses = []
        self.training_step_mae = []
        self.training_step_rmse = []

        self.validation_step_losses = []
        self.validation_step_mae = []
        self.validation_step_rmse = []

        self.weights = weights
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gamma = gamma

        self.num_features = num_features
        self.num_channels = num_channels
        self.edge_distance = edge_distance

        self.model_name = model_name
        self.batch_size = batch_size

        self.conv_reshape = []
        # models ###################
        if self.model_name == "edgecnn":
            self.model = self._get_model()

        # metrics
        self.mae = MeanAbsoluteError()
        self.mse = MeanSquaredError()

        self.save_hyperparameters()

    def loss_fn(self, output, target):
        # 🪟 Clip the target values to be within the range [0, 1]

        # unsure why it is clipped in the notebook hence commenting 
        clipped_target = torch.clip(target, min=0, max=1)
        # 📉 Calculate the mean squared error loss
        mses = torch.nn.functional.mse_loss(output, clipped_target, reduction='mean')
        return mses
        
    def _get_model(self):
        model = EdgeCNN(
            in_channels=self.num_features,  # 📊 Input features determined by the dataset
            hidden_channels=self.num_channels,  # 🕳️ Number of hidden channels in the model
            num_layers=4,  # 🧱 Number of layers in the model
            out_channels=1  # 📤 Number of output channels
        )
        return model

    def _metrics(self, y_pred, y_true):
        mae = self.mae(y_pred, y_true)
        rmse = torch.sqrt(self.mse(y_pred, y_true))
        return mae, rmse

    def  forward(self, inputs):
        output = self.model(inputs.x, inputs.edge_index)
        return output
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        
        y_pred = self.forward(batch)
        y_pred = torch.squeeze(y_pred)

        loss = self.loss_fn(y_pred[batch.valid_mask], batch.y[batch.valid_mask])
        mae, rmse = self._metrics(y_pred[batch.valid_mask], batch.y[batch.valid_mask])

        self.training_step_losses.append(loss.detach().cpu())
        self.training_step_mae.append(mae.detach().cpu())
        self.training_step_rmse.append(rmse.detach().cpu())

        self.log("train_loss", loss, prog_bar=True, on_step=True, batch_size=self.batch_size)
        self.log("train_mae", mae, prog_bar=True, on_step=True, batch_size=self.batch_size)
        self.log("train_rmse", rmse, prog_bar=True, on_step=True, batch_size=self.batch_size)

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
        y_pred = self.forward(batch)
        y_pred = torch.squeeze(y_pred)
     
        loss = self.loss_fn(y_pred[batch.valid_mask], batch.y[batch.valid_mask])
        mae, rmse = self._metrics(y_pred[batch.valid_mask], batch.y[batch.valid_mask])

        self.validation_step_losses.append(loss.detach().cpu())
        self.validation_step_mae.append(mae.detach().cpu())
        self.validation_step_rmse.append(rmse.detach().cpu())

        self.log("val_loss", loss, prog_bar=True, on_step=True, batch_size=self.batch_size)
        self.log("val_mae", mae, prog_bar=True, on_step=True, batch_size=self.batch_size)
        self.log("val_rmse", rmse, prog_bar=True, on_step=True, batch_size=self.batch_size)

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
        y_pred = self.forward(batch)
        y_pred = torch.squeeze(y_pred)
=======
import torch
import pytorch_lightning as pl
from torchmetrics.regression import MeanAbsoluteError
from torch_geometric.nn.models import EdgeCNN
import numpy as np

class GNNTrainer(pl.LightningModule):
    def __init__(
        self,
        model_name,
        num_channels,
        num_features,
        batch_size,
        edge_distance=4,
        weights=None,
        learning_rate=0.001,
        weight_decay=1e-4,
        gamma=2,
        
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

        self.num_features = num_features
        self.num_channels = num_channels
        self.edge_distance = edge_distance

        self.model_name = model_name
        self.batch_size = batch_size

        self.conv_reshape = []
        # models ###################
        if self.model_name == "edgecnn":
            self.model = self._get_model()


        # metrics ###################
        self.mae = MeanAbsoluteError()

        # self.f1 = F1Score(task=self.classification, num_classes=self.num_classes)
        # self.auroc = AUROC(task=self.classification, num_classes=self.num_classes)
        # self.recall = Recall(task=self.classification, num_classes=self.num_classes)
        # self.accuracy = Accuracy(task=, num_classes=self.num_classes)
        # self.precision = Precision(task=self.classification, num_classes=self.num_classes)
        # self.specifity = Specificity(task=self.classification, num_classes=self.num_classes)
        self.save_hyperparameters()

    def loss_fn(self, output, target):
        # 🪟 Clip the target values to be within the range [0, 1]
        clipped_target = torch.clip(target, min=0, max=1)
        # 📉 Calculate the mean squared error loss
        
        mses = torch.nn.functional.l1_loss(output, clipped_target, reduction='mean')
        #mses = torch.nn.functional.l1_loss(output, target, reduction='mean')
        return mses
        
    def _get_model(self):
        model = EdgeCNN(
            in_channels=self.num_features,  # 📊 Input features determined by the dataset
            hidden_channels=self.num_channels,  # 🕳️ Number of hidden channels in the model
            num_layers=4,  # 🧱 Number of layers in the model
            out_channels=1  # 📤 Number of output channels
        )

        return model


    def _metrics(self, y_pred, y_true):
        mae = self.mae(y_pred, y_true)
    #     f1 = self.f1(y_pred, y_true)
    #     auroc = self.auroc(y_pred, y_true)
    #     recall = self.recall(y_pred, y_true)
    #     precision = self.precision(y_pred, y_true)
    #     accuracy = self.accuracy(y_pred, y_true)
    #     specifity = self.specifity(y_pred, y_true)
        return mae

    def  forward(self, inputs):
        output = self.model(inputs.x, inputs.edge_index)
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
        inputs = batch
        

        y_pred = self.forward(inputs)
        y_pred = torch.squeeze(y_pred)
        #print(mask.unique())
        #print(y_pred.unique())       
        # loss
    
        # y_true = y_true.unsqueeze(-1)
        # y_true = y_true * mask.unsqueeze(-1) 
        loss = self.loss_fn(y_pred[batch.valid_mask], batch.y[batch.valid_mask])
        mae = self._metrics(y_pred[batch.valid_mask], batch.y[batch.valid_mask])

        self.training_step_losses.append(loss)
        # self.training_step_accuracy.append(accuracy)

        self.log("train_loss", loss, prog_bar=True, on_step=True, batch_size=self.batch_size)
        self.log("mae", mae, prog_bar=True, on_step=True, batch_size=self.batch_size)

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
        inputs = batch

        y_pred = self.forward(inputs)
        y_pred = torch.squeeze(y_pred)
     
    
        # loss
        # y_true = y_true.unsqueeze(-1)
        # y_true = y_true * mask.unsqueeze(-1) 
        loss = self.loss_fn(y_pred[batch.valid_mask], batch.y[batch.valid_mask])

        # accuracy = self._metrics(y_pred, y_true)
        mae = self._metrics(y_pred[batch.valid_mask], batch.y[batch.valid_mask])


        self.validation_step_losses.append(loss)
        # self.validation_step_accuracy.append(accuracy)

        self.log("val_loss", loss, prog_bar=True, on_step=True, batch_size=self.batch_size)
        self.log("mae", mae, prog_bar=True, on_step=True, batch_size=self.batch_size)


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

    def predict_step(self, batch, batch_idx):
        inputs = batch
        y_pred = self.forward(inputs)
        y_pred = torch.squeeze(y_pred)
        # ids = np.append(ids, batch.ids.detach().cpu().numpy())
        # preds = np.append(preds, out)

>>>>>>> f0626352f5a9962f0ea066b2438003070df20328
        return y_pred