import math
import os
from collections import OrderedDict, defaultdict
from logging import getLogger

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers.optimization import get_scheduler

from genrec.evaluator import Evaluator
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from genrec.utils import config_for_log, get_file_name, get_total_steps, log


class Trainer:
    def __init__(self, config: dict, model: AbstractModel, tokenizer: AbstractTokenizer):
        self.config = config
        self.model = model
        self.accelerator = config["accelerator"]
        self.evaluator = Evaluator(config, tokenizer)
        self.logger = getLogger()

        ckpt_dir = os.path.join(config["ckpt_dir"], config["dataset"], config["model"])
        os.makedirs(ckpt_dir, exist_ok=True)
        self.saved_model_ckpt = os.path.join(ckpt_dir, get_file_name(config, suffix=".pth"))

    def _save_model_state(self):
        model = self.accelerator.unwrap_model(self.model)
        torch.save(model.state_dict(), self.saved_model_ckpt)

    def fit(self, train_dataloader, val_dataloader):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
        )
        total_steps = get_total_steps(self.config, train_dataloader)
        if total_steps == 0:
            self.log("No training steps needed.")
            return

        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=self.config["warmup_steps"],
            num_training_steps=total_steps,
        )
        self.model, optimizer, train_dataloader, val_dataloader, scheduler = self.accelerator.prepare(
            self.model,
            optimizer,
            train_dataloader,
            val_dataloader,
            scheduler,
        )
        self.accelerator.init_trackers(
            project_name=get_file_name(self.config),
            config=config_for_log(self.config),
            init_kwargs={"tensorboard": {"flush_secs": 60}},
        )

        steps_per_epoch = max(1, len(train_dataloader) * self.accelerator.num_processes)
        epochs = int(math.ceil(total_steps / steps_per_epoch))
        best_epoch = 0
        best_val_score = -float("inf")
        global_step = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            batch_count = 0
            progress = tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc=f"Training - [Epoch {epoch + 1}]",
                disable=not self.accelerator.is_main_process,
            )
            for batch in progress:
                batch_count += 1
                optimizer.zero_grad()
                outputs = self.model(batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                if self.config["max_grad_norm"] is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                global_step += 1
                if global_step >= total_steps:
                    break

            train_loss = total_loss / max(1, batch_count)
            self.accelerator.log({"Loss/train_loss": train_loss}, step=epoch + 1)
            self.log(f"[Epoch {epoch + 1}] Train Loss: {train_loss}")

            if (epoch + 1) % self.config["eval_interval"] == 0:
                val_results = self.evaluate(val_dataloader, split="val")
                if self.accelerator.is_main_process:
                    for key, value in val_results.items():
                        self.accelerator.log({f"Val_Metric/{key}": value}, step=epoch + 1)
                    self.log(f"[Epoch {epoch + 1}] Val Results: {val_results}")

                val_metric = self.config["val_metric"]
                if val_metric not in val_results:
                    raise KeyError(f"val_metric={val_metric} is missing from validation results.")
                val_score = val_results[val_metric]
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_epoch = epoch + 1
                    if self.accelerator.is_main_process:
                        self._save_model_state()
                        self.log(f"[Epoch {epoch + 1}] Saved model checkpoint to {self.saved_model_ckpt}")

                patience = self.config["patience"]
                if patience is not None and epoch + 1 - best_epoch >= patience:
                    self.log(f"Early stopping at epoch {epoch + 1}")
                    break

            if global_step >= total_steps:
                break

        if best_epoch == 0 and self.accelerator.is_main_process:
            self._save_model_state()
            self.log(f"Saved model checkpoint to {self.saved_model_ckpt}")
        self.accelerator.wait_for_everyone()
        self.log(f"Best epoch: {best_epoch}, Best val score: {best_val_score}")

    def evaluate(self, dataloader, split="test"):
        self.model.eval()
        all_results = defaultdict(list)
        progress = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Eval - {split}",
            disable=not self.accelerator.is_main_process,
        )
        model = self.accelerator.unwrap_model(self.model)

        for batch in progress:
            batch = {key: value.to(self.accelerator.device) for key, value in batch.items()}
            with torch.inference_mode():
                preds = model.generate(batch, n_return_sequences=self.evaluator.maxk)
            labels = batch["labels"]
            if self.config["use_ddp"]:
                preds, labels = self.accelerator.gather_for_metrics((preds, labels))
            batch_results = self.evaluator.calculate_metrics(preds, labels)
            for key, value in batch_results.items():
                all_results[key].append(value)

        output = OrderedDict()
        for key in sorted(all_results):
            output[key] = torch.cat(all_results[key]).mean().item()
        return output

    def end(self):
        self.accelerator.end_training()

    def log(self, message, level="info"):
        return log(message, self.accelerator, self.logger, level=level)
