import torch
import torch.nn as nn
from copy import deepcopy
from bhaang.Medical_imaging.logger import TextLogger, CSVLogger
from bhaang.Medical_imaging.model import Model_master
from bhaang.dataset.medmnist.read_data import pytorch_dataset
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from torch.utils.data import DataLoader
import numpy as np
import time



torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)


class tent(nn.Module):
    def __init__(self, model, optimizer, tol=1e-6, max_steps=10, logger=None, experiment_logger=None, loss_fn= "Entropy"): 
        super(tent, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.tol = tol
        self.max_steps = max_steps
        if loss_fn not in ["Entropy", "MMD", "MMD_linear"]:
            raise ValueError("loss_fn must be either 'Entropy' or 'MMD' or 'MMD_linear'")
        loss_fn_dict = {
            "Entropy": self.entropy,
            "MMD": self.mmd,
            "MMD_linear": self.linear_time_mmd
        }
        self.loss_fn = loss_fn_dict[loss_fn]
        
        # Initialize loggers
        if logger is None:
            self.logger = TextLogger('logs/tent_log.txt')
        else:
            self.logger = logger
        if experiment_logger is None:
            self.experiment_logger = CSVLogger('logs/tent_experiment_log.csv')
        else:
            self.experiment_logger = experiment_logger
        
        self.logger.log("Initializing Tent adaptation module.")
        self.copy()
    
    def entropy(self, probs):
        probs = torch.clamp(probs, min=1e-10, max=1.0)
        entropy_value = -torch.sum(probs * torch.log(probs), dim=1)
        return entropy_value
    
    def mmd(self, probs, sigma=1.0):

        if hasattr(self, 'source_probs'):
            ref = self.source_probs  
        else:
            raise ValueError("source_probs not set. Please set source_probs before calling mmd.")
        
        def efficient_rbf_kernel(x, y, batch_size=512):
            """Compute RBF kernel efficiently in batches"""
            n_x = x.size(0)
            n_y = y.size(0)
            
            result = 0.0
            
            for i in range(0, n_x, batch_size):
                end_i = min(i + batch_size, n_x)
                x_batch = x[i:end_i]
                
                for j in range(0, n_y, batch_size):
                    end_j = min(j + batch_size, n_y)
                    y_batch = y[j:end_j]
                    
                    diff = x_batch.unsqueeze(1) - y_batch.unsqueeze(0)
                    dist_sq = torch.sum(diff ** 2, dim=2)
                    
                    k_batch = torch.exp(-dist_sq / (2 * sigma ** 2))
                    
                    result += k_batch.sum() / (n_x * n_y)
            
            return result
        k_tt = efficient_rbf_kernel(probs, probs)
        k_rr = efficient_rbf_kernel(ref, ref)
        k_tr = efficient_rbf_kernel(probs, ref)
        mmd_squared = k_tt + k_rr - 2 * k_tr
        
        return mmd_squared
    
    def linear_time_mmd(self, probs, sigma=1.0):
        if hasattr(self, 'source_probs'):
            ref = self.source_probs  
        else:
            raise ValueError("source_probs not set. Please set source_probs before calling mmd.")
        
        n_target = probs.size(0)
        n_source = ref.size(0)
        
        n = min(n_target, n_source)
        
        if n_target > n:
            idx = torch.randperm(n_target)[:n]
            probs = probs[idx]
        elif n_source > n:
            idx = torch.randperm(n_source)[:n]
            ref = ref[idx]
        idx = torch.randperm(n)
        half_size = n // 2
        
        k1 = probs[:half_size]
        k2 = probs[half_size:2*half_size]
        k3 = ref[:half_size]
        k4 = ref[half_size:2*half_size]
        
        def rbf(x, y):
            return torch.exp(-torch.sum((x - y) ** 2, dim=1) / (2 * sigma ** 2))
        k1k2 = rbf(k1, k2)
        k3k4 = rbf(k3, k4)
        k1k3 = rbf(k1, k3)
        k2k4 = rbf(k2, k4)
    
        mmd_est = torch.mean(k1k2) + torch.mean(k3k4) - torch.mean(k1k3) - torch.mean(k2k4)
        
        return mmd_est
    
    def freeze(self):
        # Freeze all layers except BatchNorm layers
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
                for param in module.parameters():
                    param.requires_grad = True  # Keep BN trainable
            else:
                for param in module.parameters():
                    param.requires_grad = False  # Freeze others
    
    def copy(self):
        self.model_state_dict = deepcopy(self.model.state_dict())
        
    def reset(self):
        with torch.no_grad():
            self.model.load_state_dict(self.model_state_dict)
        self.optimizer.zero_grad()
        self.freeze()
    
    @torch.enable_grad()
    def forward(self, X):
        self.reset()  # Reset model to the original state
        prev_loss = float('inf')
        self.model.train()
        for i in range(self.max_steps):
            outputs = self.model(X)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            curr_loss = torch.mean(self.loss_fn(probs))
            self.optimizer.zero_grad()
            curr_loss.backward()
            self.optimizer.step()
            # Check if improvement is below tolerance
            if (prev_loss - curr_loss) / prev_loss < self.tol:
                break
            prev_loss = curr_loss
        self.model.eval()
        return self.model(X)
    
    def test(self, target_dataloader, source_dataloader, loss_fn, experiment_name='test', train_source='organamnist'):
        # Determine device from model parameters.
        device = next(self.model.parameters()).device

        # --- Compute original accuracy on the source domain ---
        self.model.load_state_dict(self.model_state_dict)
        self.model.eval()
        original_preds_all = []
        original_true_all = []
        probs = []
        for X_source, y_source in source_dataloader:
            X_source = X_source.to(device)
            y_source = y_source.to(device)
            outputs = self.model(X_source)
            preds = torch.argmax(outputs, dim=1)
            probs.append(torch.nn.functional.softmax(outputs, dim=1).detach())
            original_preds_all.append(preds)
            original_true_all.append(y_source.squeeze(1))
        original_preds_all = torch.cat(original_preds_all)
        original_true_all = torch.cat(original_true_all)
        original_accuracy = (original_preds_all == original_true_all).float().mean().item() * 100
        self.source_probs = torch.cat(probs).detach() 
        

        self.freeze()
        self.copy()

        # --- Adapt on target domain and collect predictions ---
        target_preds_all = []
        target_true_all = []
        batch_losses = []
        batch_entropies = []
        for X_target, y_target in target_dataloader:
            X_target = X_target.to(device)
            y_target = y_target.to(device)
            outputs_target = self.forward(X_target)
            preds_target = torch.argmax(outputs_target, dim=1)
            target_preds_all.append(preds_target)
            target_true_all.append(y_target.squeeze(1))
            
            # Compute batch loss and average entropy.
            batch_loss = loss_fn(outputs_target, y_target.squeeze(1)).item()
            batch_losses.append(batch_loss)
            probs_target = torch.nn.functional.softmax(outputs_target, dim=1)
            batch_entropy = self.loss_fn(probs_target).mean().item()
            batch_entropies.append(batch_entropy)
            
        target_preds_all = torch.cat(target_preds_all)
        target_true_all = torch.cat(target_true_all)
        target_accuracy = (target_preds_all == target_true_all).float().mean().item() * 100
        test_loss = np.mean(batch_losses)
        avg_entropy = np.mean(batch_entropies)
        
        # Compute F1 and recall using sklearn.
        y_true_np = target_true_all.cpu().numpy()
        y_pred_np = target_preds_all.cpu().numpy()
        f1 = f1_score(y_true_np, y_pred_np, average='macro')
        recall = recall_score(y_true_np, y_pred_np, average='macro')
        precision = precision_score(y_true_np, y_pred_np, average='macro')
        
        # --- Evaluate adapted model on source domain using the last adapted model ---
        start_time = time.time()
        self.model.eval()
        adapted_preds_all = []
        adapted_true_all = []
        for X_source, y_source in source_dataloader:
            X_source = X_source.to(device)
            y_source = y_source.to(device)
            outputs_source = self.model(X_source)
            preds_source = torch.argmax(outputs_source, dim=1)
            adapted_preds_all.append(preds_source)
            adapted_true_all.append(y_source.squeeze(1))
        end_time = time.time()
        self.logger.log(f"Adapted model inference time: {end_time - start_time:.4f} seconds for {len(source_dataloader)} batches.")
        adapted_preds_all = torch.cat(adapted_preds_all)
        adapted_true_all = torch.cat(adapted_true_all)
        adapted_accuracy = (adapted_preds_all == adapted_true_all).float().mean().item() * 100

        # Calculate forgetting as the drop in performance on the source domain.
        forgetting = original_accuracy - adapted_accuracy

        # Reset the model back to its original state.
        self.reset()
        
        # Log experiment metrics to the experiment logger.
        metrics = {
            'experiment_name': experiment_name,
            'train_source': train_source,
            'accuracy': target_accuracy,
            'f1': f1,
            'recall': recall,
            'precision': precision,
            'loss': test_loss,
            'avg_loss (entropy or MMD)': avg_entropy,
            'forgetting': forgetting,
            'original_accuracy': original_accuracy,
            'adapted_accuracy': adapted_accuracy
        }
        self.experiment_logger.log(metrics)
        self.experiment_logger.save()
        metrics.pop('experiment_name')
        metrics.pop('train_source')
        
        return metrics


if __name__ == "__main__":

    log_file = "logs/tent_log.txt"
    logger = TextLogger(log_file)
    logger.log("Starting Tent adaptation main program.")

    model_obj = Model_master('local')
    model = model_obj.get_model('resnet_model.pth', 'resnet18')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logger.log("Model loaded and moved to device: {}.".format(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    adaptation = tent(model, optimizer, tol=1e-6, max_steps=10, logger=logger)

    # Load target domain test dataset for adaptation evaluation
    target_dataset = pytorch_dataset('test', 'organcmnist', download=True, as_rgb=True, size=64)
    target_loader = DataLoader(target_dataset, batch_size=128, shuffle=False)

    # Load source domain test dataset for computing the forgetting metric
    source_dataset = pytorch_dataset('test', 'organamnist', download=True, as_rgb=True, size=64)
    source_loader = DataLoader(source_dataset, batch_size=128, shuffle=False)

    # Evaluate the adapted model by passing both the target and source dataloaders
    metrics = adaptation.test(target_loader, source_loader, criterion, experiment_name='organcmnist test', train_source='organamnist')

    # Print out the computed metrics
    print("Test Metrics after Tent Adaptation:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    logger.log("Test evaluation finished. Metrics printed.")