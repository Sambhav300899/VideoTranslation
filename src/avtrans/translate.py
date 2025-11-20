import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

default_device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def helsinki_translate(texts, device=default_device):
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de").to(
        device
    )

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(
        device
    )
    outputs = model.generate(**inputs)

    translated_texts = [
        tokenizer.decode(output, skip_special_tokens=True) for output in outputs
    ]

    return translated_texts
