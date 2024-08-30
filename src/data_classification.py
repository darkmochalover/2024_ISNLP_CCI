import json
import torch
from torch.utils.data import Dataset
from .utils import make_chat

class CustomDataset(Dataset):
    def __init__(self, fname, model_type, tokenizer):
        self.inp = []
        self.trg = []
        
        if model_type == 'llama':
            PROMPT = '당신은 대화 맥락 추론 AI입니다. 주어진 대화 전체를 주의 깊게 읽고 전반적인 맥락과 화자들 간의 관계를 파악한 뒤, [Reference]에 대한 [Question]을 고려하여 가장 적절한 [Option]을 선택해야 합니다. \n 다음 단계를 따르세요: \n 전체 대화를 읽고 맥락을 파악한다. \n [Reference]에 집중한다. \n [Question]은 [Reference]에 대한 질문임을 인지한다. \n [Option] A, B, C 는 [Question]에 대한 답이다, 가장 적합한 추론을 선택한다. \n 선택의 근거를 대화 내용에서 찾아 검증한다. 최종 답변을 A, B, C 중 하나로 제시한다. \n [주의사항] : 대화 내용만을 근거로 판단하고, 과도한 추측은 피한다.' 
        elif model_type == 'mistral':
            PROMPT = '당신은 지시를 매우 잘 따르는 인공지능 비서입니다.'
        else:
            PROMPT = 'You are a helpful AI assistant. Please answer the user\'s questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'
            
        answer_dict = {
            "": None,
            "inference_1": 0,
            "inference_2": 1,
            "inference_3": 2
        }
        
        with open(fname, "r") as f:
            data = json.load(f)
        
        for example in data:
            chat = make_chat(example["input"], model_type)
            
            message = '<|system|>' + "\n" + PROMPT + "\n" + '<|user|>' + "\n" + chat + "\n" + '<|assistant|>'
                        
            source = tokenizer(
                message,
                return_tensors="pt",
            )
            
            # # data check
            # print(message)
            # print("-" * 50)
            
            self.inp.append(source['input_ids'].squeeze(0).tolist())  
            self.trg.append(answer_dict[example["output"]])

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return {"input_ids": self.inp[idx], "labels": self.trg[idx]}

class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.convert_tokens_to_ids('[PAD]')

    def __call__(self, instances):
        input_ids = [torch.tensor(instance['input_ids']) for instance in instances]  # Convert lists to tensors
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels = torch.tensor([instance['labels'] for instance in instances])

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': input_ids.ne(self.pad_token_id)
        }