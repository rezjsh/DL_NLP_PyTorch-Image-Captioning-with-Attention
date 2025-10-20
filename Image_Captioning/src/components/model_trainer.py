import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.entity.config_entity import ModelTrainingConfig
from torch.utils.data import DataLoader
from src.utils.logging_setup import logger
from src.utils.device import DEVICE

class Trainer:
    def __init__(self, config: ModelTrainingConfig, model, train_loader: DataLoader, val_loader: DataLoader=None, text_preprocessor=None):
        self.config = config
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.text_preprocessor = text_preprocessor
        self.pad_idx = self.text_preprocessor.stoi["<PAD>"] if self.text_preprocessor else None

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=max(1, self.config.early_stop_patience // 2)) # Patience can be half of early_stop_patience

        self.best_val_loss = float('inf')
        self.early_stop_counter = 0

        # Ensure model_dir and report_dir exist
        os.makedirs(self.config.model_dir, exist_ok=True)
        os.makedirs(self.config.report_dir, exist_ok=True)
        logger.info(f"Trainer initialized. Models will be saved to: {self.config.model_dir}")
        logger.info(f"Reports will be saved to: {self.config.report_dir}")

    def _train_epoch(self, epoch: int) -> float:
        '''Runs one full training epoch.'''
        self.model.train()
        total_loss = 0.0
        # Use self.train_loader
        for batch_idx, (images, captions, lengths) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch} Training")):
            if images is None: continue # Skip failed batches

            # Move data to DEVICE
            images = images.to(DEVICE)
            captions = captions.to(DEVICE)
            lengths = lengths.to(DEVICE)


            # Since the DataLoader is configured to use MyCollate, captions are already padded
            # We need targets to be (batch_size * max_len, vocab_size) and input to be (batch_size, max_len - 1)
            # The model predicts the next word, so the input is [SOS, w1, w2, ...] and the target is [w1, w2, ..., EOS/PAD]

            # Get the output from the model
            # predictions shape: (batch_size, max_len, vocab_size)
            # Pass captions[:, :-1] as the decoder input (tokens 0 to max_len-2)
            # Pass lengths as well
            predictions = self.model(images, captions[:, :-1], lengths) # Input is all tokens except the last one (no target for the last token)

            # Reshape predictions and captions for loss calculation
            # Target is all tokens from the second one (w1) to the end (EOS/PAD)
            targets = captions[:, 1:]

            # Reshape: (batch_size * max_len, vocab_size) for predictions
            # Reshape: (batch_size * max_len,) for targets
            loss = self.criterion(predictions.contiguous().view(-1, predictions.shape[-1]),
                                  targets.contiguous().view(-1))

            # Backpropagation with gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

            # Only update optimizer every gradient_accumulation_steps batches
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.config.gradient_accumulation_steps # Scale back up for reporting

        # Handle final step if gradient accumulation was incomplete
        if (batch_idx + 1) % self.config.gradient_accumulation_steps != 0:
            if self.config.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()


        avg_loss = total_loss / len(self.train_loader)
        logger.info(f"Epoch {epoch} Training Loss: {avg_loss:.4f}")
        return avg_loss

    def _validate(self, epoch: int) -> float:
        '''Runs one full validation epoch.'''
        if self.val_loader is None:
            logger.warning("Validation loader is not provided. Skipping validation.")
            return 0.0

        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            # Use self.val_loader
            for batch_idx, (images, captions, lengths) in enumerate(tqdm(self.val_loader, desc=f"Epoch {epoch} Validation")):
                if images is None: continue # Skip failed batches

                # Move data to DEVICE
                images = images.to(DEVICE)
                captions = captions.to(DEVICE)
                lengths = lengths.to(DEVICE)


                # Get the output from the model
                # Pass captions[:, :-1] as the decoder input
                # Pass lengths as well
                predictions = self.model(images, captions[:, :-1], lengths)

                # Reshape predictions and captions for loss calculation
                targets = captions[:, 1:]

                loss = self.criterion(predictions.contiguous().view(-1, predictions.shape[-1]),
                                      targets.contiguous().view(-1))

                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        logger.info(f"Epoch {epoch} Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def train(self):
        '''Main training loop'''
        train_losses = []
        val_losses = []

        for epoch in range(1, self.config.num_epochs + 1):
            # 1. Train
            train_loss = self._train_epoch(epoch)
            train_losses.append(train_loss)

            # 2. Validate (if validation loader exists)
            val_loss = 0.0
            if self.val_loader:
                val_loss = self._validate(epoch)
                val_losses.append(val_loss)
                self.scheduler.step(val_loss) # Step the scheduler based on validation loss
            else:
                self.scheduler.step(train_loss) # Fallback: Step the scheduler based on training loss

            # 3. Save Checkpoint
            is_best = False
            if self.val_loader and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                is_best = True
                self.early_stop_counter = 0
            elif self.val_loader:
                self.early_stop_counter += 1

            # Save checkpoint every N epochs or if it's the best model
            if epoch % self.config.save_every_epochs == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)

            # 4. Early Stopping Check
            if self.val_loader and self.early_stop_counter >= self.config.early_stop_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}. Validation loss did not improve for {self.config.early_stop_patience} epochs.")
                break

        # 5. Final Report
        self._save_training_report(train_losses, val_losses)
        logger.info("Training complete.")

    def save_checkpoint(self, epoch, is_best=False):
        ckpt_dir = self.config.model_dir
        os.makedirs(ckpt_dir, exist_ok=True) # Ensure directory exists

        filename = f"{self.config.model_save_prefix}_epoch_{epoch}.pth"
        if is_best:
            filename = f"{self.config.model_save_prefix}_best_model.pth"

        path = os.path.join(ckpt_dir, filename)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model checkpoint saved to {path}")

    def _save_training_report(self, train_losses, val_losses):
        report_dir = self.config.report_dir
        os.makedirs(report_dir, exist_ok=True)

        # Save losses as CSV
        df = pd.DataFrame({
            'epoch': range(1, len(train_losses) + 1),
            'train_loss': train_losses,
        })
        if val_losses:
            df['val_loss'] = val_losses

        csv_path = os.path.join(report_dir, 'training_report.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Training report saved to {csv_path}")

        # Plot losses
        plt.figure(figsize=(8,6))
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
        if val_losses:
            plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss over Epochs')
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(report_dir, 'loss_curve.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Loss curve plot saved to {plot_path}")