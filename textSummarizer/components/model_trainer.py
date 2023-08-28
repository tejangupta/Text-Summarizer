from textSummarizer.entity import ModelTrainerConfig
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, \
    EarlyStoppingCallback
from datasets import load_from_disk
import os


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.trainer = None

    def train(self, learning_rate, num_train_epochs, batch_size, weight_decay):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)

        # loading data
        dataset_samsum_pt = load_from_disk(str(self.config.data_path))

        optimizer = torch.optim.AdamW(model_pegasus.parameters(), lr=learning_rate, weight_decay=weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_train_epochs * len(dataset_samsum_pt["train"])
        )

        early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

        trainer_args = TrainingArguments(
            output_dir=str(self.config.root_dir),
            num_train_epochs=num_train_epochs,
            warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=weight_decay,
            logging_steps=self.config.logging_steps,
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=learning_rate,
            load_best_model_at_end=True
        )

        self.trainer = Trainer(model=model_pegasus,
                               args=trainer_args,
                               tokenizer=tokenizer,
                               data_collator=seq2seq_data_collator,
                               train_dataset=dataset_samsum_pt["train"],
                               eval_dataset=dataset_samsum_pt["validation"],
                               optimizers=(optimizer, lr_scheduler),
                               callbacks=[early_stopping]
                               )

        self.trainer.train()

        # Save the trained model and tokenizer
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum-model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))

    def get_validation_loss(self):
        if self.trainer:
            return self.trainer.state.best_metric
        else:
            return None
