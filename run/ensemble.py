'''
Soft Voting (각 모델들의 불러와 진행)
'''
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"

os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"

import argparse
import json
import tqdm

import torch
import numpy
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig

from peft import AutoPeftModelForSequenceClassification, PeftModel, prepare_model_for_kbit_training, LoftQConfig, LoraConfig, PrefixTuningConfig, AdaptionPromptConfig, AdaptionPromptConfig, LoHaConfig, LoKrConfig, LNTuningConfig, get_peft_model, TaskType

# from src.data import CustomDataset
from src.data_classification import CustomDataset, DataCollatorForSupervisedDataset
from unsloth import FastLanguageModel, FastLlamaModelSequenceClassification, FastSequenceClassificationModel
from unsloth import is_bfloat16_supported

from trl import SFTTrainer

import argparse
from src.arg_parser_ensemble import get_args


from unsloth import FastLanguageModel

import re 

'''
model, tokenizer 로드
'''
def load_model_tokenizer(model_path, special_tokens_dict, device):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 1024,
        dtype = torch.bfloat16,
        load_in_4bit = True,
        resize_model_vocab = special_tokens_dict,
        device_map = device,
        sequence_classification = True,
        num_labels = 3,
    )

    model.to(args.device)
    model.eval()

    return model, tokenizer

# '''
# hard voting 앙상블
# '''
# def hard_voting_ensemble(votes, model, inputs, idx):
#     with torch.no_grad():
#         output = model(inputs)
#         preds = output.logits.argmax(-1)
#         votes[idx][preds.item()] += 1
#     return votes

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
                                                         '<|option_1|>', '<|option_2|>', '<|option_3|>', '<|pad|>'
                                                         ]}
    
    
    model_paths = ['/home/nlplab/hdd1/smk/baseline/resource/model/2024-08-20-02-50/checkpoint-875',
                   '/home/nlplab/hdd1/smk/baseline/resource/model/category/전제_비단게_875/전제/checkpoint-4830']
    

    # row=len(inputs) : input 개수, col=3 : inferece 3개
    votes = torch.zeros((605, 3), dtype=torch.float32).to(args.device)

    answer_dict = {
        0: "inference_1",
        1: "inference_2",
        2: "inference_3",
    }
        
    with open("resource/data/대화맥락추론_test.json", "r") as f:
        result = json.load(f)

    
    with torch.no_grad():
        for i, model_path in enumerate(model_paths):
            model, tokenizer = load_model_tokenizer(model_path, special_tokens_dict, args.device)
            
            
            dataset = CustomDataset("resource/data/대화맥락추론_test.json", args.model_architecture, tokenizer)

            for idx in tqdm.tqdm(range(len(dataset))):
                input = dataset[idx]['input_ids'].unsqueeze(0).to(args.device)
                
                with torch.no_grad():
                    output = model(input)
                    votes[idx] += output.logits.squeeze(0)
            print(votes)

            del model
            del dataset
            del tokenizer
                
            
    print(votes)
    final_preds = votes.argmax(-1)
    for idx, pred in enumerate(final_preds):
        result[idx]["output"] = answer_dict[pred.item()]

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    args = get_args()
    exit(main(args))
