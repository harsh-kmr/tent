from torchvision.models import ResNet18_Weights
from bhaang.Medical_imaging.model import Model_master
from bhaang.dataset.medmnist.read_data import pytorch_dataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from bhaang.Medical_imaging.logger import TextLogger, CSVLogger
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import numpy as np
import time



torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)


class Trainer:
    def __init__(self, model, train_loader, val_loader=None, num_epochs=100,
                  device='cuda', optimizer=None, criterion=None, csv_logger=None, logger=None, source_dataloader_name=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = device
        self.optimizer = optimizer if optimizer else torch.optim.Adam(model.parameters(), lr=0.001)
        self.criterion = criterion if criterion else torch.nn.CrossEntropyLoss()
        self.model.to(self.device)
        self.logger = logger 
        self.csv_logger = csv_logger
        self.source_dataloader_name = source_dataloader_name

    def train(self):
        self.logger.log("Starting training")
        start_time = time.time()
        self.csv_logger.create_column('epoch')
        self.csv_logger.create_column('train_loss')
        if self.val_loader:
            self.csv_logger.create_column('val_loss')
        
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = self._train_one_epoch()
            if self.val_loader:
                val_loss = self._validate_one_epoch()
                self.logger.log(f'Epoch: {epoch+1}/{self.num_epochs}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, source: {self.source_dataloader_name}')
                self.csv_logger.log({'epoch': epoch+1, 'train_loss': train_loss, 'val_loss': val_loss, 'source': self.source_dataloader_name})
            else:
                self.logger.log(f'Epoch: {epoch+1}/{self.num_epochs}, Loss: {train_loss:.4f}, source: {self.source_dataloader_name}')
                self.csv_logger.log({'epoch': epoch+1, 'train_loss': train_loss, 'source': self.source_dataloader_name})
        self.csv_logger.save()
        self.logger.log("Training completed")
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.logger.log(f"Training time: {elapsed_time:.2f} seconds for {self.num_epochs} epochs on dataset {self.source_dataloader_name} with {len(self.train_loader.dataset)} samples")
        return self.model

    def _train_one_epoch(self):
        total_loss = 0
        for images, labels in tqdm(self.train_loader, desc="Training"):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels.squeeze(1))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def _validate_one_epoch(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels.squeeze(1))
                total_loss += loss.item()
        return total_loss / len(self.val_loader)
    
    

    def test(self, test_loader, experiment_name=None, experiment_logger=None):
        # experiment_logger is a csv logger which is used to log the results of the experiment
        start_time = time.time()
        exp_name = experiment_name if experiment_name is not None else "Test"
        self.model.eval()
        all_labels = []
        all_preds = []
        batch_losses = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Testing on {exp_name}"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)

                all_labels.extend(labels.squeeze(1).cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

                batch_loss = self.criterion(outputs, labels.squeeze(1)).item()
                batch_losses.append(batch_loss)

        # Calculate metrics using sklearn
        accuracy = accuracy_score(all_labels, all_preds) * 100
        f1 = f1_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro')
        loss = np.mean(batch_losses)

        # Log the results with the experiment_logger if provided
        if experiment_logger is not None:
            experiment_logger.log({
                'experiment_name': exp_name,
                'train_source': self.source_dataloader_name,
                'accuracy': accuracy,
                'f1': f1,
                'recall': recall,
                'precision': precision,
                'loss': loss
            })

            experiment_logger.save()
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.logger.log(f"Testing time: {elapsed_time:.2f} seconds for {len(test_loader.dataset)} samples")
        return {
            'accuracy': accuracy,
            'f1': f1,
            'recall': recall,
            'precision': precision
        }


if __name__ == '__main__':
    logger = TextLogger('train_log.txt')
    csv_logger = CSVLogger('train_log.csv')
    experiment_logger = CSVLogger('experiment_log.csv')

    train_dataset = pytorch_dataset('train', 'organamnist', download=True, as_rgb=True, size=64)
    val_dataset   = pytorch_dataset('val', 'organamnist', download=False, as_rgb=True, size=64)
    train_loader  = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader    = DataLoader(val_dataset, batch_size=128, shuffle=False)

    model_obj = Model_master('torch')
    model = model_obj.get_model('pytorch/vision', 'resnet18', weights=ResNet18_Weights.DEFAULT)
    sample_input = torch.randn(1, 3, 64, 64)
    model_obj.display_model_layers(sample_input)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    model = model_obj.add_classification_head(11)
    
    for images, labels in train_loader:
        logger.log(f'Image shape: {images.shape}, Label shape: {labels.shape}')
        logger.log(f'Labels: {labels.squeeze(1)}')
        break

    trainer = Trainer(model, train_loader, val_loader, num_epochs=10, device=device, 
                      optimizer=optimizer, criterion=criterion, csv_logger=csv_logger, logger=logger)
    model = trainer.train()
    model_obj.model = model
    model_obj.save_model('resnet_model.pth')

    def evaluate_dataset(trainer, dataset_key, download):
        """Loads train, val, and test splits for a given dataset key and evaluates the model."""
        for split in ['train', 'val', 'test']:
            ds = pytorch_dataset(split, dataset_key, download=download, as_rgb=True, size=64)
            loader = DataLoader(ds, batch_size=128, shuffle=(split == 'train'))
            trainer.test(loader, f'{dataset_key} {split}', experiment_logger)
    
    evaluate_dataset(trainer, 'organamnist', download=False)
    evaluate_dataset(trainer, 'organcmnist', download=True)
    evaluate_dataset(trainer, 'organsmnist', download=True)

    logger.log("Loading the model from the saved file")
    logger.log("")

    local_model = Model_master('local')
    local_model.get_model('resnet_model.pth', 'resnet18')
    sample_input = sample_input.to(device)
    local_model.display_model_layers(sample_input)

    local_trainer = Trainer(local_model.model, train_loader, val_loader, num_epochs=10, device=device, 
                            optimizer=optimizer, criterion=criterion, csv_logger=csv_logger, logger=logger)
    # check the model on the test data
    evaluate_dataset(local_trainer, 'organamnist', download=False)
    evaluate_dataset(local_trainer, 'organcmnist', download=True)
    evaluate_dataset(local_trainer, 'organsmnist', download=True)

    experiment_logger.save()

    # check if both models are same
    assert torch.allclose(model.fc.weight, local_model.model.fc.weight), "Model weights are not same"
    assert torch.allclose(model.fc.bias, local_model.model.fc.bias), "Model biases are not same"
