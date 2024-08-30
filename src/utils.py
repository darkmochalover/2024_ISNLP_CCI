
import torch
import random
import numpy as np
from torch import nn

from torch.nn.functional import softmax

from sklearn.metrics import accuracy_score


from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType

cate_2_sent = {
    '원인' : '대상 발화의 사건을 유발하는 사건', 
    '후행사건' : '대상 발화 이후에 일어날 수 있는 사건',
    '전제' : '대상 발화의 사건을 가능하게 하는 상태 혹은 사건', 
    '동기' : '대상 발화를 일으키는 \"화자\"의 감정이나 기본 욕구',
    '반응' : '대상 발화 사건에 대해 \'청자\'가 보일 수 있는 감정 반응'
}

def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
def lm_loss_fn(targets, outputs):
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    logits = outputs.logits.view(-1, outputs.logits.size(-1))  # [batch_size * seq_len, vocab_size]
    target = targets.view(-1)  # [batch_size * seq_len]
    
    loss = criterion(logits, target)
    return loss

def cls_loss_fn(targets, outputs):
    criterion = nn.CrossEntropyLoss()
    logits = outputs.logits
    loss = criterion(logits, targets)
    return loss

def question_tmpl(category, use_cate_2_sent=True):
    # 카테고리를 자세하게 풀이하여 사용
    # [Question] 위 대화의 *대상 발화의 사건을 유발하는 사건*에 대해 올바른 지문은?
    if use_cate_2_sent:
        question = f"[Question]\n위 대화의 {cate_2_sent[category]}에 대해 올바른 지문은?"
        
    # 카테고리를 바로 대입해 사용
    # [Question] 위 대화의 *원인*에 대해 올바른 지문은?
    else:
        question = f"[Question]\n위 대화의 {inp['category']}"
        if (ord(category[-1]) - ord("가")) % 28 > 0:
            question += "으로"
        else:
            question += "로"
        question += " 올바른 지문은?"
    
    return question
    

def make_chat(inp, model_type=None):
    if model_type == 'llama':
        chat = ["[Conversation]"]
        for cvt in inp['conversation']:
            speaker = cvt['speaker']
            utterance = cvt['utterance']
            
            chat.append(f"<|speaker_{speaker}|> 화자{speaker}: {utterance} <|speaker_{speaker}|>")
        chat = "\n".join(chat)

        question = question_tmpl(inp['category'], True)
            
        chat = chat + "\n\n" + question + "\n\n[Option]\n"
        chat += f"<|option_1|> A. {inp['inference_1']} <|option_1|>\n"
        chat += f"<|option_2|> B. {inp['inference_2']} <|option_2|>\n"
        chat += f"<|option_3|> C. {inp['inference_3']} <|option_3|>"
        
    else:
        chat = ["[Conversation]"]
        for cvt in inp['conversation']:
            speaker = cvt['speaker']
            utterance = cvt['utterance']
            chat.append(f"화자{speaker}: {utterance}")
        chat = "\n".join(chat)

        question = question_tmpl(inp['category'], True)
            
        chat = chat + "\n\n" + question + "\n\n[Option]\n"
        chat += f"A. {inp['inference_1']}\n"
        chat += f"B. {inp['inference_2']}\n"
        chat += f"C. {inp['inference_3']}"

    return chat
