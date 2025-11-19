import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.entity.config_entity import ModelTrainingConfig
from src.utils.logging_setup import logger 
from src.utils.device import DEVICE 
from torch.amp import autocast, GradScaler

class Trainer:
    def __init__(self, config: ModelTrainingConfig, model, train_loader: DataLoader, val_loader: DataLoader=None, text_preprocessor=None):
        self.config = config
        self.model = model.to(DEVICE)
        if DEVICE.type == 'cuda' and hasattr(torch, 'compile'):
            # Compiling the model for maximum performance. 'bwd' is often the most stable mode.
            self.model = torch.compile(self.model, mode="reduce-overhead")
            logger.info("Model successfully compiled using torch.compile for performance.")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.text_preprocessor = text_preprocessor

        # Ensure text_preprocessor is provided if needed
        if text_preprocessor and hasattr(self.text_preprocessor, "stoi"):
            self.pad_idx = self.text_preprocessor.stoi.get("<PAD>", -1) # Use .get() and fallback
        else:
             self.pad_idx = 0 # Default to 0 or another safe value if vocab isn't set up yet

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        # Use max(1, ...) to ensure patience is at least 1
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=max(1, self.config.early_stop_patience // 2), factor=0.5, threshold=0.0001)

        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        self.scaler = GradScaler('cuda')
        # Ensure model_dir and report_dir exist
        os.makedirs(self.config.model_dir, exist_ok=True)
        os.makedirs(self.config.report_dir, exist_ok=True)
        logger.info(f"Trainer initialized. Models will be saved to: {self.config.model_dir}")
        logger.info(f"Reports will be saved to: {self.config.report_dir}")

    def _train_epoch(self, epoch: int) -> float:
        '''Runs one full training epoch with gradient accumulation and clipping.'''
        self.model.train()
        total_loss = 0.0

        num_batches = len(self.train_loader)

        # Ensure gradients are cleared at the start of the epoch
        self.optimizer.zero_grad()
        use_amp = self.scaler is not None

        for batch_idx, (images, captions, lengths) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch} Training")):
            if images is None:
                logger.warning(f"Skipping empty batch {batch_idx+1}.")
                continue

            # Move data to DEVICE
            images = images.to(DEVICE)
            if torch.isnan(images).any():
              logger.error("NaN detected in image inputs!")
              raise ValueError("Input NaN")
            # Decoder input sequence (excluding the last token, which has no target)
            decoder_input = captions[:, :-1].to(DEVICE)
            # Target sequence (excluding the first token, which is <SOS>)
            targets = captions[:, 1:].to(DEVICE)
            # Lengths are not always strictly necessary for a fully padded Transformer
            # but are kept to match the model's expected signature.
            lengths = lengths.to(DEVICE)
            with autocast(device_type=DEVICE.type, enabled=use_amp):
                # 1. Forward Pass: predictions shape (batch_size, max_len-1, vocab_size)
                # The model's forward pass should correctly handle the length of the decoder_input
                predictions = self.model(images, decoder_input, lengths)

                # 2. Reshape predictions and targets for CrossEntropyLoss
                # predictions view: (Batch_size * max_len-1, vocab_size)
                # targets view: (Batch_size * max_len-1,)
                loss = self.criterion(predictions.contiguous().view(-1, predictions.shape[-1]),
                                      targets.contiguous().view(-1))

                # 3. Scale loss for accumulation and Backpropagation
                accumulation_factor = self.config.gradient_accumulation_steps
                loss_scaled = loss / accumulation_factor
            if use_amp:
                # Scaled backward pass
                self.scaler.scale(loss_scaled).backward()
            else:
                # Normal backward pass
                loss_scaled.backward()


            # 4. Accumulate UN-SCALED loss for accurate reporting (using the actual batch loss)
            total_loss += loss.item() # loss.item() is the un-scaled loss for the current batch

            # 5. Optimizer Step
            # Only update optimizer every `accumulation_factor` batches
            if (batch_idx + 1) % accumulation_factor == 0:
                if self.config.clip_grad_norm > 0:
                    if use_amp:
                        self.scaler.unscale_(self.optimizer) # Unscale gradients before clipping
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # 5. Accumulate un-scaled loss for reporting
            # Note: We use the batch loss (before scaling) to report the average loss per batch
            # total_loss += loss.item() * accumulation_factor

        # 6. Handle final step if gradient accumulation was incomplete
        if (batch_idx + 1) % accumulation_factor != 0:
            if self.config.clip_grad_norm > 0:
                if use_amp:
                  # Unscale gradients before clipping and stepping
                  self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)

            if use_amp:
                # Optimizer step using scaler
                self.scaler.step(self.optimizer)
                self.scaler.update() # Update the scaler for the next iteration
            else:
                # Normal optimizer step
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
        use_amp = self.scaler is not None

        with torch.no_grad():
            for batch_idx, (images, captions, lengths) in enumerate(tqdm(self.val_loader, desc=f"Epoch {epoch} Validation")):
                if images is None: continue

                images = images.to(DEVICE)
                decoder_input = captions[:, :-1].to(DEVICE)
                targets = captions[:, 1:].to(DEVICE)
                lengths = lengths.to(DEVICE)
                with autocast(device_type=DEVICE.type, enabled=use_amp):
                  # Get the output from the model
                  predictions = self.model(images, decoder_input, lengths)

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
            val_loss = None
            if self.val_loader:
                val_loss = self._validate(epoch)
                val_losses.append(val_loss)
                self.scheduler.step(val_loss) # Step scheduler on val loss
            else:
                self.scheduler.step(train_loss) # Fallback to train loss

            # 3. Checkpoint & Early Stopping Logic
            is_best = False
            current_loss = val_loss if val_loss is not None else train_loss

            if current_loss < self.best_val_loss:
                self.best_val_loss = current_loss
                is_best = True
                self.early_stop_counter = 0
                logger.info(f"New best model found! Loss: {current_loss:.4f}")
            elif self.val_loader: # Only count early stop if validation is active
                self.early_stop_counter += 1
                logger.info(f"Early stopping counter: {self.early_stop_counter}/{self.config.early_stop_patience}")

            # Save checkpoint every N epochs or if it's the best model
            if epoch % self.config.save_every_epochs == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)

            # 4. Early Stopping Check
            if self.val_loader and self.early_stop_counter >= self.config.early_stop_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}. Validation loss did not improve.")
                break

        # 5. Final Report
        self._save_training_report(train_losses, val_losses)
        logger.info("Training complete.")

    def save_checkpoint(self, epoch, is_best=False):
        ckpt_dir = self.config.model_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        filename = f"{self.config.model_save_prefix}_epoch_{epoch}.pth"
        if is_best:
            filename = f"{self.config.model_save_prefix}_best_model.pth"

        path = os.path.join(ckpt_dir, filename)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model checkpoint saved to {path}")

    def _save_training_report(self, train_losses, val_losses):
        report_dir = self.config.report_dir
        os.makedirs(report_dir, exist_ok=True)
        num_epochs = len(train_losses)

        # Pad val_losses with NaN/None to match the length of train_losses.
        val_loss_padded = val_losses + [None] * (num_epochs - len(val_losses))

        # Save losses as CSV
        df = pd.DataFrame({
            'epoch': range(1, num_epochs + 1),
            'train_loss': train_losses,
            'val_loss': val_loss_padded,
        })
        # # Align lengths of val_losses with train_losses for DataFrame
        # if val_losses:
        #     num_val_epochs = len(val_losses)
        #     # Create a Series with NaNs for missing epochs if val_loader was not used initially
        #     val_series = pd.Series([None] * len(train_losses))
        #     val_series[-num_val_epochs:] = val_losses
        #     df['val_loss'] = val_series.tolist()

        csv_path = os.path.join(report_dir, 'training_report.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Training report saved to {csv_path}")

        # Plot losses
        plt.figure(figsize=(8,6))
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
       # Use .dropna() to plot only the epochs that had a validation step
        if not all(v is None for v in val_loss_padded):
            plt.plot(df['epoch'][df['val_loss'].notna()], df['val_loss'].dropna(), label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss over Epochs')
        plt.legend()
        plt.grid(True)


        plot_path = os.path.join(report_dir, 'loss_curve.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Loss curve plot saved to {plot_path}")