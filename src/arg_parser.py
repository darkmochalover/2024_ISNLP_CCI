import argparse

from datetime import datetime, timezone, timedelta

def get_args():
    
    timestamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d-%H-%M")
    
    parser = argparse.ArgumentParser(prog="train", description="Training about Conversational Context Inference.")
    
    g = parser.add_argument_group("Common Parameter")
    g.add_argument("--model_id", type=str, required=True, help="model file path")
    
    g.add_argument("--tokenizer", type=str, help="huggingface tokenizer path")
    g.add_argument("--device", type=str, required=True, help="device to load the model")
    g.add_argument("--epoch", type=int, default=5, help="training epoch")
    g.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    g.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")
    g.add_argument("--save_dir", type=str, default=f"./resource/model/{timestamp}", help="model save path")
    
    g.add_argument("--batch_size", type=int, default=1, help="batch size")
    g.add_argument("--seed", type=int, default=42, help="random seed")
    g.add_argument("--scheduler_type", type=str, default="cosine", help="scheduler type")
    g.add_argument("--warmup_steps", type=int, default=20, help="warmup steps")
    g.add_argument("--weight_decay", type=float, default=0.01, help="weight decay")
    g.add_argument("--save_total_limit", type=int, default=5, help="save total limit")
    g.add_argument("--max_seq_len", type=int, default=1024, help="max sequence length")
    
    g.add_argument("--model_task_type", type=str, default="casual_lm")
    g.add_argument("--model_architecture", type=str, default="decoder")
    
    g.add_argument("--max_train_steps", type=int, default=1000000)
    g.add_argument("--parallel_mode", type=str, default="dist_flash_attn")
    g.add_argument("--model_scaler", type=float, default=0.2, help="model scaler - expert adapter 사용률")
    # g.add_argument("--max_train_steps", type=int, default=1000000)
    
    g.add_argument("--train_file", type=str, default="resource/data/train.json", help="train file path")
    g.add_argument("--valid_file", type=str, default="resource/data/dev.json", help="valid file path")
    g.add_argument("--test_file", type=str, default="resource/data/test.json", help="test file path")

    g.add_argument("--visualize_outputs", type=bool, default=False, help="visualize outputs")
    
    g = parser.add_argument_group("Custom Model Mode Parameter")
    g.add_argument("--model_custom_type", type= str, default="max_pooling", help="Custom model type")
    g.add_argument("--token_scale_alpha", type= float, default=0.2, help="alpha value")
    
    g = parser.add_argument_group("Prompt Parameter")
    g.add_argument("--prompt", type= str, default="You are a helpful AI assistant. Please answer the user\'s questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.", help="Prompt for the model")
    
    g = parser.add_argument_group("Quantization Parameter")
    g.add_argument("--do_quantization", type=str2bool, nargs='?', const=True, default=False, help="quantization flag")
    g.add_argument("--quantization", action="store_true", help="quantization flag")
    g.add_argument("--4bit", action="store_true", help="4bit quantization flag")
    
    g = parser.add_argument_group("Quantization Parameter")
    g.add_argument("--do_peft", type=str2bool, nargs='?', const=True, default=False, help="PEFT (T) / Full fine-tuning (F) flag")
    g.add_argument("--peft_type", type=str, default='lora', help="lora / prefix-tuning / ...")
    
    g = parser.add_argument_group("Prefix-tuning Parameter")
    g.add_argument("--num_virtual_tokens", type=int, default=20, help="virtual tokens length")
    
    g = parser.add_argument_group("LoRA Parameter")
    g.add_argument("--lora_r", type=int, default=8, help="lora r value")
    g.add_argument("--lora_alpha", type=int, default=16, help="lora alpha value")
    g.add_argument("--lora_dropout", type=float, default=0.05, help="lora dropout value")
    
    g = parser.add_argument_group("loftq Parameter")
    g.add_argument("--use_loftq", type=str2bool, nargs='?', const=True, default=False, help="PEFT (T) / Full fine-tuning (F) flag")
    g.add_argument("--loftq_bits", type=int, default=8, help="loftq bits value")
        
    g = parser.add_argument_group("Wandb Options")
    g.add_argument("--log_wandb", type=str2bool, nargs='?', const=False, default=True, help="wandb log or not")
    g.add_argument("--wandb_run_name", type=str, default=timestamp, help="wandb run name")
    g.add_argument("--wandb_project_name", type=str, default="CCI_2024", help="wandb project name")
    g.add_argument("--wandb_entity", type=str, default="darkmochalover", help="wandb entity name")
    
    g = parser.add_argument_group("Inference Options")
    g.add_argument("--output", type=str, default="./default_output.json")
    g.add_argument("--saved_peft_weight", type=str, default="../resource/model/best/epoch_14_accuracy_0.8344/")
 
 
    g = parser.add_argument_group("Inference Strategy")
    g.add_argument("--num_beams", type=int, default=1, help="num beams")
    g.add_argument("--top_k", type=int, default=50, help="top k")
    g.add_argument("--top_p", type=float, default=1.0, help="top p")
    
    args = parser.parse_args()
    return args

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
