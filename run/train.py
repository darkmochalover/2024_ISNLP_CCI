

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["TORCH_USE_CUDA_DSA"] = '1'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["GRADIO_SHARE"]="1"
os.environ["WORLD_SIZE"] = "1"

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"

import argparse
from src.arg_parser import get_args

import numpy as np

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
# from trl import SFTTrainer, SFTConfig

from unsloth import FastLanguageModel, FastLlamaModelSequenceClassification
from unsloth import is_bfloat16_supported

from trl import SFTTrainer
from transformers import TrainingArguments

# from src.data_classification import CustomDataset, DataCollatorForSupervisedDataset
# from src.data_casual import CustomDataset, DataCollatorForSupervisedDataset
from src.data_classification_attn import CustomDataset, DataCollatorForSupervisedDataset

from src.utils import get_peft_config, load_tokenizer

from peft import prepare_model_for_kbit_training, LoftQConfig, LoraConfig, PrefixTuningConfig, AdaptionPromptConfig, AdaptionPromptConfig, LoHaConfig, LoKrConfig, LNTuningConfig, get_peft_model, TaskType

from accelerate import infer_auto_device_map, init_empty_weights

import wandb

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# os.environ['WORLD_SIZE'] = '1'
# os.environ['RANK'] = '1'
# os.environ['LOCAL_RANK'] = '1'

# os.environ['NCCL_DEBUG'] = 'INFO' 
# # os.environ["NCCL_DEBUG"] = 'WARN'

# os.environ['NCCL_P2P_DISABLE']='1' 
# os.environ['NCCL_IB_DISABLE']='1' 
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["TORCH_USE_CUDA_DSA"] = '0, 1'
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


# os.environ["WANDB_LOG_MODEL"] = "checkpoint"

torch.cuda.init()  # CUDA 시스템 초기화
# print(torch.cuda.is_available())  # CUDA 디바이스 사용 가능 여부 확인
# torch.cuda.reset_peak_memory_stats(device="cuda:1")  # 모든 디바이스에 대해 메모리 통계 재설정

if torch.cuda.is_available():
    print("Available CUDA Device Count: ", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print("Device ", i, ": ", torch.cuda.get_device_name(i))
        
# import torch.distributed as dist

# torch.distributed.init_process_group(backend='nccl')

from sklearn.metrics import accuracy_score

# 카테고리별로 모델학습
# inference -> category -> 카테고리별 모델
       
def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    # print("\n\n[preprocess_logits_for_metrics]")
    # print("logits: ", logits)
    # print("logits: ", logits[0])
    # assert 0
    pred_ids = torch.argmax(logits[0], dim=-1)
    # print("pred_ids: ", pred_ids)
    # print("labels: ", labels)
    # assert 0
    return pred_ids, labels


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # preds = pred.predictions
    accuracy = (preds  == labels).sum().item() / len(labels)

    return {
        'accuracy': accuracy,
    }  
    
def main(args):
    special_tokens_dict = {'additional_special_tokens': ['<|system|>', '<|user|>', '<|assistant|>',
                                                         '<|speaker_1|>', '<|speaker_2|>', 
                                                         '<|option_1|>', '<|option_2|>', '<|option_3|>',
                                                         '<|pad|>'
                                                         ]}
    
    
    # initial model, optimizer, dataloader and acclerator
    quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
                )
    
    if args.use_loftq:
        loftq_config = LoftQConfig(loftq_bits=args.loftq_bits,) 
    
    # Model, Tokenizer load
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_id,
        max_seq_length = args.max_seq_len,
        dtype = torch.bfloat16,
        load_in_4bit = True,
        resize_model_vocab = special_tokens_dict,
        device_map = args.device,
        sequence_classification = True,
        num_labels = 3,
        load_custom_model = False,
    )
    
    # print("model: ", model)
    
    if args.do_quantization:
        model = prepare_model_for_kbit_training(model)
    
    # Do model patching and add fast LoRA weights
    model = FastLlamaModelSequenceClassification.get_peft_model(
        model,
        r = args.lora_r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = args.lora_alpha,   
        lora_dropout = args.lora_dropout,                   # Supports any, but = 0 is optimized
        bias = "none",                                      # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 42,
        max_seq_length = args.max_seq_len,
        # use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = loftq_config if args.use_loftq else None, # And LoftQ
    )
    
    special_tokens_dict = {'additional_special_tokens': ['<|system|>', '<|user|>', '<|assistant|>',
                                                         '<|speaker_1|>', '<|speaker_2|>', 
                                                         '<|option_1|>', '<|option_2|>', '<|option_3|>', '<|pad|>']}
    
    
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.pad_token =  '<|pad|>' 
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<|pad|>')
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model.resize_token_embeddings(len(tokenizer))
           
    # 데이터셋 불러오기.
    train_dataset = CustomDataset(args.train_file, args.model_architecture, tokenizer)
    valid_dataset = CustomDataset(args.valid_file, args.model_architecture, tokenizer)
    
    
    train_dataset = Dataset.from_dict({
        'input_ids': train_dataset.inp,
        # 'attention_mask': train_dataset.att,
        # 'token_type_ids': train_dataset.token_type,
        "labels": train_dataset.trg,
        })
    valid_dataset = Dataset.from_dict({
        'input_ids': valid_dataset.inp,
        # 'attention_mask': valid_dataset.att,
        # 'token_type_ids': valid_dataset.token_type,
        "labels": valid_dataset.trg,
        })
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    
    trainer = Trainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = valid_dataset,
        
        compute_metrics = compute_metrics,
        # preprocess_logits_for_metrics = preprocess_logits_for_metrics,
        
        args = TrainingArguments(
            do_train = True,
            do_eval = True,
            
            per_device_train_batch_size = 1,
            per_device_eval_batch_size = 1,
            gradient_accumulation_steps = 4,
            eval_accumulation_steps = 4,
            
            torch_empty_cache_steps = 1,
            warmup_steps = args.warmup_steps,
            num_train_epochs = args.epoch, # Set this for 1 full training run.
            # evaluation_strategy='epoch',
            
            # max_steps = 50,
            eval_strategy = "steps",
            eval_steps = 50,
            
            learning_rate =  args.lr,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = args.weight_decay,
            lr_scheduler_type = args.scheduler_type,
            seed = 3407,
            output_dir = args.save_dir,
            save_strategy="steps",
            save_steps=150,
            load_best_model_at_end=True,
            report_to="wandb" if args.log_wandb else 'none',
        ),
    )

    trainer.train()
    
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
    exit(main(args))
    
