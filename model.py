from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

model = AutoModelForMaskedLM.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")