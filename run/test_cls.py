import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["XFORMERS_FORCE_DISABLE_TRITON"]="1"

import argparse
import json
import tqdm

import torch
import numpy
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig

from peft import AutoPeftModelForSequenceClassification, PeftModel, prepare_model_for_kbit_training, LoftQConfig, LoraConfig, PrefixTuningConfig, AdaptionPromptConfig, AdaptionPromptConfig, LoHaConfig, LoKrConfig, LNTuningConfig, get_peft_model, TaskType

# from src.data import CustomDataset
from src.data_classification_prompt import CustomDataset, DataCollatorForSupervisedDataset
from unsloth import FastLanguageModel, FastLlamaModelSequenceClassification, FastSequenceClassificationModel
from unsloth import is_bfloat16_supported

from trl import SFTTrainer

import argparse
from src.arg_parser import get_args


from unsloth import FastLanguageModel

import re

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    accuracy = (preds  == labels).sum().item() / len(labels)

    return {
        'accuracy': accuracy,
    }  
    
def main(args):
    # initial model, optimizer, dataloader and acclerator
    quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
                )
    
    # if args.use_loftq:
    #     loftq_config = LoftQConfig(loftq_bits=args.loftq_bits,) 
        
    special_tokens_dict = {'additional_special_tokens': ['<|system|>', '<|user|>', '<|assistant|>',
                                                         '<|speaker_1|>', '<|speaker_2|>', 
                                                         '<|option_1|>', '<|option_2|>', '<|option_3|>',
                                                         '<|pad|>'
                                                         ]}
    
    
    # unsloth으로 load (현재 안됨)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.saved_peft_weight, # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length =  args.max_seq_len,
        dtype = torch.bfloat16,
        load_in_4bit = True,
        resize_model_vocab = special_tokens_dict,
        device_map = args.device,
        sequence_classification = True,
        num_labels = 3,
    )
    # tokenizer.pad_token = tokenizer.eos_token
    
    # if args.do_quantization:
    #     model = prepare_model_for_kbit_training(model)
        
    # FastLlamaModelSequenceClassification.for_inference(model) # Enable native 2x faster inferenceå
    
    dataset = CustomDataset(args.test_file, args.model_architecture, tokenizer)

    answer_dict = {
        0: "inference_1",
        1: "inference_2",
        2: "inference_3",
    }
        
    with open("resource/data/test.json", "r") as f:
        result = json.load(f)

    model.to(args.device)
    model.eval()
    
    with torch.no_grad():
        for idx in tqdm.tqdm(range(len(dataset))):
            inp = dataset[idx]['input_ids']
            outputs = model(inp.unsqueeze(0),)
            
            preds = outputs.logits.argmax(-1)
            
            result[idx]["output"] = answer_dict[preds[0].item()]

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))
    
    # 정답지와 비교
    with open("resource/data/gold.json", "r") as dev:
        golds = json.load(dev)
        
    filename = args.output.split("/")[-1].split(".")[0]
    
    acc = 0
    idx = 0
    failed_idx = []
    
    for gold in golds:
        if gold['output'] == result[idx]['output']:
            acc += 1
        else:
            failed_idx.append(idx)
        idx += 1
    
    avg_acc = acc / len(golds)
    
    print("\n\n[정답지 비교 결과]")
    print("acc: ", avg_acc)
    print("failed_idx: ", failed_idx)
    
    with open(f"{filename}.txt", "w", encoding="utf-8") as record:
        record.write(f"acc: {avg_acc}\n")
        record.write(f"failed_idx: {failed_idx}\n")


if __name__ == "__main__":
    args = get_args()
    exit(main(args))
    
    
    
