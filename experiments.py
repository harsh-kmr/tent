from tent import tent
from bhaang.Medical_imaging.model import Model_master
from bhaang.dataset.medmnist.read_data import pytorch_dataset
from bhaang.Medical_imaging.logger import CSVLogger, TextLogger
from noise import CTNoisyTransform
import torchvision.transforms as transforms
from train import Trainer
from torchvision.models import ResNet18_Weights
from merged_datasets import merge_datasets

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import time



logger = TextLogger('experiment_log.txt')
train_logger = CSVLogger('train_log.csv')
tent_logger = CSVLogger('tent_log.csv')
train_experiment_logger = CSVLogger('train_experiment_log.csv')

transform_pipeline = transforms.Compose([
    transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),  # Convert input to a tensor if necessary.
    CTNoisyTransform(apply_blur=True, blur_kernel_size=5, blur_sigma=1.0,
                     apply_gaussian_noise=True, gaussian_std=0.05)
    # To add Poisson noise instead, set `apply_poisson_noise=True` and adjust parameters.
])

noises = {
    "gaussian": CTNoisyTransform(
        apply_blur=False,
        apply_gaussian_noise=True, gaussian_mean=0.0, gaussian_std=0.05,
        apply_poisson_noise=False
    ),
    "poisson": CTNoisyTransform(
        apply_blur=False,
        apply_gaussian_noise=False,
        apply_poisson_noise=True, poisson_scale=10.0
    ),
    "blur": CTNoisyTransform(
        apply_blur=True, blur_kernel_size=5, blur_sigma=1.0,
        apply_gaussian_noise=False,
        apply_poisson_noise=False
    )
}

test_datasets = {
    "organamnist": pytorch_dataset('test', 'organamnist', download=False, as_rgb=True, size=64, transform=None),
    "organcmnist": pytorch_dataset('test', 'organcmnist', download=False, as_rgb=True, size=64, transform=None),
    "organsmnist": pytorch_dataset('test', 'organsmnist', download=False, as_rgb=True, size=64, transform=None),
}

train_datasets = {
    "organamnist": pytorch_dataset('train', 'organamnist', download=False, as_rgb=True, size=64, transform=None),
    "organcmnist": pytorch_dataset('train', 'organcmnist', download=False, as_rgb=True, size=64, transform=None),
    "organsmnist": pytorch_dataset('train', 'organsmnist', download=False, as_rgb=True, size=64, transform=None),
}

val_datasets = {
    "organamnist": pytorch_dataset('val', 'organamnist', download=False, as_rgb=True, size=64, transform=None),
    "organcmnist": pytorch_dataset('val', 'organcmnist', download=False, as_rgb=True, size=64, transform=None),
    "organsmnist": pytorch_dataset('val', 'organsmnist', download=False, as_rgb=True, size=64, transform=None),
}

Merged_Datasets = {
    "organamnist": merge_datasets('organamnist', splits=["train", "val", "test"], download=False, as_rgb=True, size=64, transform=None),
    "organcmnist": merge_datasets('organcmnist', splits=["train", "val", "test"], download=False, as_rgb=True, size=64, transform=None),
    "organsmnist": merge_datasets('organsmnist', splits=["train", "val", "test"], download=False, as_rgb=True, size=64, transform=None),
    "organamnist_blured": merge_datasets('organamnist', splits=["train", "val", "test"], download=False, as_rgb=True, size=64, transform=noises["blur"]),
    "organamnist_gaussian": merge_datasets('organamnist', splits=["train", "val", "test"], download=False, as_rgb=True, size=64, transform=noises["gaussian"]),
    "organamnist_poisson": merge_datasets('organamnist', splits=["train", "val", "test"], download=False, as_rgb=True, size=64, transform=noises["poisson"]),
    "organcmnist_blured": merge_datasets('organcmnist', splits=["train", "val", "test"], download=False, as_rgb=True, size=64, transform=noises["blur"]),
    "organcmnist_gaussian": merge_datasets('organcmnist', splits=["train", "val", "test"], download=False, as_rgb=True, size=64, transform=noises["gaussian"]),
    "organcmnist_poisson": merge_datasets('organcmnist', splits=["train", "val", "test"], download=False, as_rgb=True, size=64, transform=noises["poisson"]),
    "organsmnist_blured": merge_datasets('organsmnist', splits=["train", "val", "test"], download=False, as_rgb=True, size=64, transform=noises["blur"]),
    "organsmnist_gaussian": merge_datasets('organsmnist', splits=["train", "val", "test"], download=False, as_rgb=True, size=64, transform=noises["gaussian"]),
    "organsmnist_poisson": merge_datasets('organsmnist', splits=["train", "val", "test"], download=False, as_rgb=True, size=64, transform=noises["poisson"])
}
logger.log("Merged datasets loaded successfully.")


dataset_loaders = {}
for dataset_name, dataset in Merged_Datasets.items():
    dataset_loaders[dataset_name] = DataLoader(dataset, batch_size=128, shuffle=True)

test_loaders = {}
for dataset_name, dataset in test_datasets.items():
    test_loaders[dataset_name] = DataLoader(dataset, batch_size=128, shuffle=False)

train_loaders = {}
for dataset_name, dataset in train_datasets.items():
    train_loaders[dataset_name] = DataLoader(dataset, batch_size=128, shuffle=True)

val_loaders = {}
for dataset_name, dataset in val_datasets.items():
    val_loaders[dataset_name] = DataLoader(dataset, batch_size=128, shuffle=False)

modalities = ["organamnist", "organcmnist", "organsmnist"]
noise_keys = {
    "original": "",
    "blur": "_blured",
    "gaussian": "_gaussian",
    "poisson": "_poisson"
}
noise_types = list(noise_keys.keys())

n_rows = len(modalities)
n_cols = len(noise_types)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))

for row, modality in enumerate(modalities):
    for col, noise in enumerate(noise_types):
        key = modality + noise_keys[noise]
        loader = DataLoader(Merged_Datasets[key], batch_size=1, shuffle=False)
        for images, labels in loader:
            image = images[0]
            label = labels[0].item()
            break

        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        if image.shape[0] == 1:
            ax.imshow(image.squeeze(0).numpy(), cmap='gray')
        else:
            ax.imshow(image.permute(1, 2, 0).numpy())
            
        ax.set_title(f"{modality}{noise_keys[noise]} \n{noise} - Label: {label}")
        ax.axis('off')

plt.tight_layout()
plt.savefig('example_images.png')
#plt.show()

def create_model(dropout=0):
    model_obj = Model_master('torch')
    model = model_obj.get_model('pytorch/vision', 'resnet18', weights=ResNet18_Weights.DEFAULT)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model = model_obj.add_classification_head(11)
    model_obj.model = model
    if dropout > 0:
        model = model_obj.add_dropout(dropout)
        model_obj.model = model
    model_obj.display_model_layers(torch.randn(1, 3, 64, 64))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    model_obj.model = model
    return model_obj, optimizer, criterion


models = {
    "organamnist": create_model(),
    "organcmnist": create_model(),
    "organsmnist": create_model()
}


def train_model(model_obj, optimizer, criterion, train_loader, val_loader, n_epochs=10, logger=logger, source_dataloader_name=None):
    logger.log(f"Training model with {n_epochs} epochs.")
    logger.log(f"source_dataloader_name: {source_dataloader_name}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_obj.model
    model.to(device)
    trainer = Trainer(model, train_loader, val_loader, num_epochs=n_epochs, device=device, optimizer=optimizer, criterion=criterion, csv_logger=train_logger, logger=logger, source_dataloader_name= source_dataloader_name)
    model = trainer.train()

    return model, trainer

logger.log("Training models on the original datasets.")
logger.log("")

trained_models = {
    "organamnist": train_model(*models["organamnist"], train_loader=train_loaders["organamnist"], val_loader=val_loaders["organamnist"], source_dataloader_name="organamnist"),
    "organcmnist": train_model(*models["organcmnist"], train_loader=train_loaders["organcmnist"], val_loader=val_loaders["organcmnist"], source_dataloader_name="organcmnist"),
    "organsmnist": train_model(*models["organsmnist"], train_loader=train_loaders["organsmnist"], val_loader=val_loaders["organsmnist"], source_dataloader_name="organsmnist")
}



logger.log("")
logger.log("Models trained successfully.")

def evaluate_model(model, trainer, dataset_loaders):
    for key, loader in dataset_loaders.items():
        trainer.test(loader, f"{key}", train_experiment_logger)

logger.log("Evaluating the trained models on the test datasets.")
logger.log("")
for key, (model, trainer) in trained_models.items():
    evaluate_model(model, trainer, dataset_loaders)

logger.log("")
logger.log("Models evaluated successfully.")

logger.log("testing the tent adaptation")

criterion = torch.nn.CrossEntropyLoss()

for modality in modalities:
    for key, loader in dataset_loaders.items():
        if key == modality:
            continue
        trained_model = trained_models[modality][0]
        trained_trainer = trained_models[modality][1]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trained_model.to(device)
        adaptation_optimizer = torch.optim.Adam(trained_model.parameters(), lr=0.001)
        loss_fn = "Entropy"
        start_time = time.time()
        logger.log(f"Adapting model from {modality} to {key} using TENT using {loss_fn} loss.")
        tent_adaptation = tent(trained_model, adaptation_optimizer, tol=1e-6, max_steps=10, logger=logger, experiment_logger=tent_logger, loss_fn=loss_fn)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.log(f"TENT adaptation time: {elapsed_time:.2f} seconds for {len(loader.dataset)} samples")
        start_time = time.time()
        metrics = tent_adaptation.test(loader, test_loaders[modality], criterion, experiment_name=f"{key}", train_source=modality)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.log(f"TENT testing time: {elapsed_time:.2f} seconds for {len(loader.dataset)} samples")

# save the experiment log
tent_logger.save()
train_logger.save()
train_experiment_logger.save()
logger.log("Experiment logs saved successfully.")