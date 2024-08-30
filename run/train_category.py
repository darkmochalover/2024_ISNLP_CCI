import os
import sys
import argparse
import torch
import wandb
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from unsloth import FastLanguageModel, FastLlamaModelSequenceClassification, is_bfloat16_supported
from peft import prepare_model_for_kbit_training, LoftQConfig

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["NCCL_P2P_DISABLE"] = "0"
os.environ["NCCL_IB_DISABLE"] = "0"

os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "0"

from src.data_per_option import CustomDataset, DataCollatorForSupervisedDataset
from src.arg_parser import get_args


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy}

def train_category(args, model, tokenizer, category):
    
    train_dataset = CustomDataset(args.train_file, args.model_architecture, tokenizer, category)
    valid_dataset = CustomDataset(args.valid_file, args.model_architecture, tokenizer, category)
    
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        args=TrainingArguments(
            output_dir=f"{args.save_dir}/{category}",
            num_train_epochs=args.epoch,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            warmup_steps=5,
            learning_rate=args.lr,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            evaluation_strategy="steps",
            eval_steps=125,
            save_strategy="steps",
            save_steps=125,
            load_best_model_at_end=True,
            report_to="wandb" if args.log_wandb else None,
        ),
    )
    
    trainer.train()
    
    trainer.save_model(f"{args.save_dir}/{category}")
    
    eval_result = trainer.evaluate()
    return eval_result

def main(args):
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map={'':torch.cuda.current_device()},
        sequence_classification=True,
        num_labels=3,
    )
    
    if args.do_quantization:
        model = prepare_model_for_kbit_training(model)
    
    model = FastLlamaModelSequenceClassification.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        max_seq_length=args.max_seq_len,
        loftq_config=LoftQConfig(loftq_bits=args.loftq_bits) if args.use_loftq else None,
    )
    
    special_tokens_dict = {'additional_special_tokens': ['<|system|>', '<|user|>', '<|assistant|>',
                                                         '<|speaker_1|>', '<|speaker_2|>', 
                                                         '<|option_1|>', '<|option_2|>', '<|option_3|>', '<|pad|>', '<|reference|>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.pad_token = '<|pad|>'
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<|pad|>')
    tokenizer.padding_side = 'left'
    model.resize_token_embeddings(len(tokenizer))
    
    categories = [args.category]
    
    for category in categories:
        print(f"Training for category: {category}")
        if args.log_wandb:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                name=f"{args.wandb_run_name}_{category}",
                config=args,
            )
        
        eval_result = train_category(args, model, tokenizer, category)
        print(f"Evaluation result for {category}: {eval_result}")
        
        if args.log_wandb:
            wandb.finish()

if __name__ == "__main__":
    args = get_args()
    main(args)