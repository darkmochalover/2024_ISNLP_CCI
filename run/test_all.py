import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["XFORMERS_FORCE_DISABLE_TRITON"]="1"

import json
import tqdm

import torch
from transformers import BitsAndBytesConfig

from unsloth import FastLanguageModel, FastLlamaModelSequenceClassification, FastSequenceClassificationModel
from unsloth import is_bfloat16_supported

import argparse
from src.arg_parser import get_args
from src.model_test import model_test, model_util1_test, model_util2_test, hard_voting


from unsloth import FastLanguageModel

def main(args):
    
    # 원인
    cause = model_util2_test("resource/model/호이야_1250", "원인", args.device)
    total = cause
    del cause
    
    # 동기
    motive = model_util2_test("resource/model/아이야_2125", "동기", args.device)
    total += motive
    del motive
    
    # 반응
    response = model_util2_test("resource/model/아이야_2125", "반응", args.device)
    total += response
    del response
    
    # 전제
    premise_1 = model_util2_test("resource/model/아이야_1250", "전제", args.device) 
    premise_2 = model_util2_test("resource/model/비단게_전제_1375", "전제", args.device) 
    premise_3 = model_util1_test("resource/model/비단게_875", "전제", args.device)
    premise = hard_voting(premise_1, premise_2, premise_3)
    total += premise
    del premise, premise_1, premise_2, premise_3
    
    # # 후행사건
    subsequent_1 = model_test("resource/model/바닐라1_5000", "후행사건", args.device)
    subsequent_2 = model_test("resource/model/납작복숭아2_5000", "후행사건", args.device) 
    subsequent_3 = model_util2_test("resource/model/아이야_2125", "후행사건", args.device)
    subsequent = hard_voting(subsequent_1, subsequent_2, subsequent_3)
    total += subsequent
    del subsequent, subsequent_1, subsequent_2, subsequent_3
    
    
    answer_dict = {
        0: "inference_1",
        1: "inference_2",
        2: "inference_3",
    }
    
    with open("resource/data/대화맥락추론_test.json", "r") as f:
        result = json.load(f)
    
    final_preds = total.argmax(-1)
    for idx, pred in enumerate(final_preds):
        result[idx]["output"] = answer_dict[pred.item()]

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))

if __name__ == "__main__":
    args = get_args()
    exit(main(args))