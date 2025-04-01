"""
Trainer class for fine-tuning language models.
"""
import os
import logging
from typing import Dict, List, Optional, Union, Any

import torch
import transformers
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

logger = logging.getLogger(__name__)


class LLMTrainer:
    """Class for fine-tuning LLMs using PEFT methods."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM trainer.
        
        Args:
            config: Configuration dictionary with training parameters
        """
        self.config = config
        self.model_name = config.get("model_name", "mistralai/Mistral-7B-Instruct-v0.2")
        self.output_dir = config.get("output_dir", "./models/finetuned")
        self.dataset_name = config.get("dataset_name")
        self.dataset_path = config.get("dataset_path")
        
        # Training parameters
        self.lora_r = config.get("lora_r", 8)
        self.lora_alpha = config.get("lora_alpha", 16)
        self.lora_dropout = config.get("lora_dropout", 0.05)
        self.use_4bit = config.get("use_4bit", True)
        self.bnb_4bit_compute_dtype = config.get("bnb_4bit_compute_dtype", "float16")
        self.bnb_4bit_quant_type = config.get("bnb_4bit_quant_type", "nf4")
        self.use_nested_quant = config.get("use_nested_quant", False)
        
        # Set up model, tokenizer, and dataset
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.dataset = self._load_dataset()
        
        # Set up training arguments
        self.training_args = self._setup_training_args()
        self.trainer = self._setup_trainer()
    
    def _load_tokenizer(self):
        """Load the tokenizer."""
        logger.info(f"Loading tokenizer for {self.model_name}")
        return AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
    
    def _load_model(self):
        """Load the model with quantization and LoRA."""
        logger.info(f"Loading model {self.model_name}")
        
        # Quantization config
        if self.use_4bit:
            compute_dtype = getattr(torch, self.bnb_4bit_compute_dtype)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.use_nested_quant,
            )
        else:
            quantization_config = None
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Add LoRA adapter
        if self.use_4bit:
            model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        model = get_peft_model(model, lora_config)
        return model
    
    def _load_dataset(self):
        """Load and prepare the dataset."""
        logger.info("Loading dataset")
        
        if self.dataset_name:
            # Load from Hugging Face hub
            dataset = load_dataset(self.dataset_name)
        elif self.dataset_path:
            # Load from local file
            dataset = load_dataset("json", data_files=self.dataset_path)
        else:
            raise ValueError("Either dataset_name or dataset_path must be provided")
        
        # Tokenize dataset
        tokenize_function = lambda examples: self._tokenize_function(examples)
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        return tokenized_dataset
    
    def _tokenize_function(self, examples):
        """Tokenize the examples."""
        # Format the prompt-response pairs
        formatted_texts = []
        for i in range(len(examples["prompt"])):
            prompt = examples["prompt"][i]
            response = examples["response"][i]
            formatted_text = f"<s>[INST] {prompt} [/INST] {response} </s>"
            formatted_texts.append(formatted_text)
        
        # Tokenize
        return self.tokenizer(
            formatted_texts,
            padding="max_length",
            truncation=True,
            max_length=self.config.get("max_length", 512),
            return_tensors="pt"
        )
    
    def _setup_training_args(self):
        """Setup training arguments."""
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.config.get("num_epochs", 3),
            per_device_train_batch_size=self.config.get("batch_size", 4),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 2),
            optim=self.config.get("optimizer", "adamw_torch"),
            save_steps=self.config.get("save_steps", 100),
            logging_steps=self.config.get("logging_steps", 10),
            learning_rate=self.config.get("learning_rate", 2e-4),
            weight_decay=self.config.get("weight_decay", 0.001),
            fp16=self.config.get("fp16", True),
            bf16=self.config.get("bf16", False),
            max_grad_norm=self.config.get("max_grad_norm", 0.3),
            max_steps=self.config.get("max_steps", -1),
            warmup_ratio=self.config.get("warmup_ratio", 0.03),
            group_by_length=self.config.get("group_by_length", True),
            lr_scheduler_type=self.config.get("lr_scheduler_type", "cosine"),
        )
    
    def _setup_trainer(self):
        """Setup the trainer."""
        # Use DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        return Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset["train"],
            data_collator=data_collator
        )
    
    def train(self):
        """Train the model."""
        logger.info("Starting training")
        self.trainer.train()
        
        # Save the final model
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"Training completed. Model saved to {self.output_dir}")
        
    def evaluate(self, eval_dataset=None):
        """Evaluate the model."""
        if eval_dataset is None and "validation" in self.dataset:
            eval_dataset = self.dataset["validation"]
        
        if eval_dataset is None:
            logger.warning("No evaluation dataset provided")
            return None
        
        logger.info("Starting evaluation")
        eval_results = self.trainer.evaluate(eval_dataset=eval_dataset)
        
        logger.info(f"Evaluation results: {eval_results}")
        return eval_results


if __name__ == "__main__":
    # Example configuration
    config = {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
        "output_dir": "./models/finetuned-mistral",
        "dataset_path": "./data/processed/training_data.json",
        "num_epochs": 3,
        "batch_size": 4,
        "learning_rate": 2e-4,
        "max_length": 512
    }
    
    # Create trainer and train
    trainer = LLMTrainer(config)
    trainer.train()

