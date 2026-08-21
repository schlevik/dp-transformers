from transformers import AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")


dataset_path = "./sampled_n2c2"


max_length = 0
for i in range(100):
    file_path = f"{dataset_path}/dataset_{i}.jsonl"
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            text = data["text"]
            tokens = tokenizer(text, return_tensors="pt")['input_ids'][0]
            print(len(tokens))
            max_length = max(max_length, len(tokens))
    
    print("max_length", max_length)



print("*"*100)
print("final max length: ", max_length)
print("*"*100)