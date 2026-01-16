import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import random


class InMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform):
        self.images = []
        self.labels = []
        for path, label in samples:
            img = Image.open(path).convert('RGB')
            t = transform(img)
            self.images.append(t)
            self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def main():
    # enhanced transforms with augmentation to reduce overfitting
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomGrayscale(p=0.1),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # validation/test transforms - no augmentation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # preload all images into RAM to avoid disk I/O during training
    raw_folder = ImageFolder(root='data/')
    samples = raw_folder.samples  # list of (path, label)
    num_classes = len(raw_folder.classes)
    print(f"Number of classes: {num_classes}")

    full_dataset = InMemoryDataset(samples, train_transform)

    train_size = int(0.7 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # DataLoader params: use 0 workers on Windows to avoid spawn issues
    is_windows = os.name == 'nt'
    num_workers = 2 if is_windows else 4

    loader_kwargs = {"batch_size": 16, "shuffle": True, "num_workers": num_workers}
    val_loader_kwargs = {"batch_size": 4, "shuffle": False, "num_workers": num_workers}
    if device.type == "cuda":
        loader_kwargs.update({"pin_memory": True})
        val_loader_kwargs.update({"pin_memory": True})

    train_loader = DataLoader(train_dataset, **loader_kwargs)
    val_loader = DataLoader(val_dataset, **val_loader_kwargs)

    weights = models.ResNet152_Weights.IMAGENET1K_V2
    model = models.resnet152(weights=weights)
    
    # # unfreeze later layers for fine-tuning (improve feature learning)
    # for param in model.layer3.parameters():
    #     param.requires_grad = True
    # for param in model.layer4.parameters():
    #     param.requires_grad = True
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # add dropout for regularization
    class ResNet152WithDropout(nn.Module):
        def __init__(self, base_model, num_classes):
            super().__init__()
            self.base = base_model
            
        def forward(self, x):
            x = self.base(x)
            return x
    
    model = ResNet152WithDropout(model, num_classes)
    model.to(device)

    # use cross-entropy loss
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # criterion = nn.CosineEmbeddingLoss()
    
    # optimizer with learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
    
    num_epochs = 150
    best_val_acc = 0.0
    patience = 8
    patience_counter = 0

    # prepare lists for plotting
    train_losses = []
    train_accs = []
    train_precisions = []
    train_recalls = []
    val_losses = []
    val_accs = []
    val_precisions = []
    val_recalls = []
    sample_img = None

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        all_train_preds = []
        all_train_labels = []
        
        for inputs, labels in train_loader:
            # capture a sample input (unnormalized) for visualization
            sample = inputs[0].cpu()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = sample.numpy().transpose(1, 2, 0)
            img = (img * std) + mean
            img = np.clip(img, 0, 1)
            sample_img = img
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model.base(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()
            
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct_preds / total_preds if total_preds > 0 else 0.0
        epoch_precision = precision_score(all_train_labels, all_train_preds, average='weighted', zero_division=0)
        epoch_recall = recall_score(all_train_labels, all_train_preds, average='weighted', zero_division=0)
        
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        train_precisions.append(epoch_precision)
        train_recalls.append(epoch_recall)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Prec: {epoch_precision:.4f} Recall: {epoch_recall:.4f}")

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model.base(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                
        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_correct / val_total if val_total > 0 else 0.0
        val_epoch_precision = precision_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)
        val_epoch_recall = recall_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)
        
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)
        val_precisions.append(val_epoch_precision)
        val_recalls.append(val_epoch_recall)
        print(f"Val   Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f} Prec: {val_epoch_precision:.4f} Recall: {val_epoch_recall:.4f}")
        
        # early stopping
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            patience_counter = 0
            torch.save(model.state_dict(), "twitter_classifier_best.pth")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # plot progress: left = sample image, right = loss/accuracy curves
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            if sample_img is not None:
                axes[0, 0].imshow(sample_img)
                axes[0, 0].axis('off')
                axes[0, 0].set_title('Sample input')

            # Loss curves
            epochs = np.arange(1, len(train_losses) + 1)
            axes[0, 1].plot(epochs, train_losses, label='train loss', marker='o')
            axes[0, 1].plot(epochs, val_losses, label='val loss', marker='x')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].set_title('Loss Progress')

            # Accuracy curves
            axes[1, 0].plot(epochs, train_accs, label='train acc', marker='o')
            axes[1, 0].plot(epochs, val_accs, label='val acc', marker='x')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].legend()
            axes[1, 0].set_title('Accuracy Progress')
            
            # Precision & Recall
            axes[1, 1].plot(epochs, train_precisions, label='train precision', marker='o')
            axes[1, 1].plot(epochs, val_precisions, label='val precision', marker='x')
            axes[1, 1].plot(epochs, train_recalls, label='train recall', marker='s', linestyle='--')
            axes[1, 1].plot(epochs, val_recalls, label='val recall', marker='^', linestyle='--')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
            axes[1, 1].set_title('Precision & Recall Progress')

            plt.tight_layout()
            plt.savefig('training_progress.png', dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"Plotting error: {e}")

    # load best model
    model.load_state_dict(torch.load("twitter_classifier_best.pth"))
    torch.save(model.state_dict(), "twitter_classifier.pth")


if __name__ == "__main__":
    main()