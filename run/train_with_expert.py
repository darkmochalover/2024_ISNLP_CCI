
import os
import sys
import shutil
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
from src.arg_parser import get_args

import numpy as np
import random
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from unsloth import FastLanguageModel, FastLlamaModelSequenceClassification
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from src.data_classification_prompt import CustomDataset, DataCollatorForSupervisedDataset
from src.utils import get_peft_config, load_tokenizer
from peft import prepare_model_for_kbit_training, LoftQConfig
from accelerate import infer_auto_device_map, init_empty_weights
import wandb
import torch.nn as nn

import json
import tqdm

# Environment 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

os.environ["NCCL_P2P_DISABLE"] = "0"
os.environ["NCCL_IB_DISABLE"] = "0"
os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "0"

cate_2_id = {
        "원인": 0,
        "후행사건": 1, 
        "전제":  2, 
        "동기" : 3,
        "반응" : 4,
    }

# adapter_dict = {
#         "원인": "adapter_cause",
#         "후행사건": "adapter_subsequent",
#         "전제":  "adapter_prerequisite",
#         "동기" : "adapter_motivation",
#         "반응" : "adapter_emotional",
#     }

adapter_dict = {
        0 : "adapter_cause",
        1 : "adapter_subsequent",
        2 :  "adapter_prerequisite",
        3 : "adapter_motivation",
        4 : "adapter_emotional",
    }

def set_seed(seed: int):
    """Set all seeds to make results reproducible"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    

def seed_worker(worker_id):
    """Set seed for dataloader workers"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = (preds == labels).sum().item() / len(labels)
    return {'accuracy': accuracy}

class Custom_model_with_expert(nn.Module):
    def __init__(self, model, alpha=0.2):
        super(Custom_model_with_expert, self).__init__()
        self.model = model
        self.alpha = alpha                                      # scale term for the expert adapter
        

    def forward(self, input_ids, attention_mask, labels, category):
        # Expert Adapter
        self.model.set_adapter(adapter_dict[category.item()])
        outputs_from_active = self.model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                                category = category)
        
        # Default Adapter
        self.model.set_adapter('adapter_all')
        outputs_from_all = self.model(input_ids=input_ids, 
                             attention_mask=attention_mask, 
                            #  token_type_ids=token_type_ids,
                             labels=labels,
                             category=category)
        
        logit = (1-self.alpha) * outputs_from_all.logits + self.alpha * outputs_from_active.logits
        loss =  (1-self.alpha) * outputs_from_all.loss + self.alpha * outputs_from_active.loss
        
        
        return {
                "loss": loss,
                "logits": logit
            }
    
def main(args):
    
    set_seed(args.seed)
    

    # CUDA initialization
    torch.cuda.init()
    if torch.cuda.is_available():
        print("Available CUDA Device Count: ", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print("Device ", i, ": ", torch.cuda.get_device_name(i))

    # Model configuration
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_threshold=6.0,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )
    
    if args.use_loftq:
        loftq_config = LoftQConfig(loftq_bits=args.loftq_bits,)

    # Tokenizer configuration
    special_tokens_dict = {
        'additional_special_tokens': [
            '<|system|>', '<|user|>', '<|assistant|>',
            '<|speaker_1|>', '<|speaker_2|>', 
            '<|option_1|>', '<|option_2|>', '<|option_3|>', '<|pad|>', '<|reference|>'
        ]
    }
    
    # Model and Tokenizer loading
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        resize_model_vocab = special_tokens_dict,
        device_map=args.device,
        sequence_classification=True,
        num_labels=3,
    )

    if args.do_quantization:
        model = prepare_model_for_kbit_training(model)

    
    # Model patching and adding fast LoRA weights
    model = FastLlamaModelSequenceClassification.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        max_seq_length=args.max_seq_len,
        loftq_config=loftq_config if args.use_loftq else None,
    )

    
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.pad_token = '<|pad|>'
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<|pad|>')
    
    
    tokenizer.padding_side = 'left'
    model.resize_token_embeddings(len(tokenizer))
    
    # Load pretrained adapter
    pretrained_adapter={
            "adapter_all": args.adapter_all,
            "adapter_cause": args.adapter_cause,
            "adapter_subsequent": args.adapter_subsequent,
            "adapter_prerequisite": args.adapter_prerequisite,
            "adapter_motivation": args.adapter_motivation,
            "adapter_emotional": args.adapter_emotional,
    }

    for adapter_name, adapter_weight in pretrained_adapter.items():
        print("Loading adapter: ", adapter_name)
        model.load_adapter(adapter_name=adapter_name,
                            model_id=adapter_weight,
                            strict=False)
    
    # score 모듈에서 expert adapter 제거
    score_module = model.base_model.model.score
    
    for adapter_name, adapter_weight in pretrained_adapter.items():
        if adapter_name != 'adapter_all':
            del score_module.modules_to_save[adapter_name]
            
    
    model_with_expert = Custom_model_with_expert(model, args.model_scaler)    
            
    
    # Dataset preparation
    train_dataset = CustomDataset(args.train_file, args.model_architecture, tokenizer)
    valid_dataset = CustomDataset(args.valid_file, args.model_architecture, tokenizer)

    train_dataset = Dataset.from_dict({
        'input_ids': train_dataset.inp,
        "labels": train_dataset.trg,
        "category": train_dataset.category,
    })
    valid_dataset = Dataset.from_dict({
        'input_ids': valid_dataset.inp,
        "labels": valid_dataset.trg,
        "category": valid_dataset.category,
    })

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    
    # Trainer configuration
    trainer = Trainer(
        model=model_with_expert,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        args=TrainingArguments(
            do_train=True,
            do_eval=True,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            eval_accumulation_steps=16,
            torch_empty_cache_steps=32,
            warmup_steps=args.warmup_steps,
            max_steps=250,
            num_train_epochs=args.epoch,
            eval_strategy="steps",
            eval_steps=10,
            learning_rate=args.lr,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.scheduler_type,
            seed=args.seed,
            output_dir=args.save_dir,
            save_strategy="steps",
            save_steps=10,
            load_best_model_at_end=True,
            report_to="wandb" if args.log_wandb else 'none',
            dataloader_num_workers=1,
            dataloader_pin_memory=False,
        ),
        data_collator=data_collator,
    )
    
    # Set the seed worker for the dataloaders
    trainer.get_train_dataloader().worker_init_fn = seed_worker
    trainer.get_eval_dataloader().worker_init_fn = seed_worker

    trainer.train()
    
    # adapter_all 이외의 adapter가 해당 모듈에 있으면 삭제 - 모델 저장 시 adapter_all만 저장
    for name, module in model.base_model.model.named_modules():
        if isinstance(module, (nn.ModuleDict, nn.ParameterDict)):
            for adapter_name, adapter_weight in pretrained_adapter.items():
                if adapter_name != 'adapter_all':
                    if adapter_name in module:
                        del module[adapter_name]
            
    trainer.save_model(args.save_dir)


if __name__ == "__main__":
    args = get_args()
    
    if args.log_wandb:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=args,
        )
    else:
        os.environ["WANDB_MODE"] = "disabled"
    exit(main(args))