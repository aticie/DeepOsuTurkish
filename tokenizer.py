from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "unsloth/Qwen3-1.7B-Base-unsloth-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name)

new_tokens = [
    "ım","im","um","üm",
    "ın","in","un","ün",
    "ımız","imiz","umuz","ümüz",
    "ınız","iniz","unuz","ünüz",
    "nin","nın","nun","nün",
    "si","sı","su","sü",
    "leri","ları",
]

added = tokenizer.add_tokens(new_tokens, special_tokens=False)
print("Added:", added)

if added > 0:
    model.resize_token_embeddings(len(tokenizer))
print(tokenizer)

def show(text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    toks = tokenizer.convert_ids_to_tokens(ids)
    return toks, len(ids)

for w in ["evim", "evimiz", "kitabınız", "arkadaşımın", "gözlüğümüz", "Türkiye'deki"]:
    print(w, show(w))