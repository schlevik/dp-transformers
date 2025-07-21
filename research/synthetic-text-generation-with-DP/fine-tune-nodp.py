# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Train GPT2 model series without DP (w/ parameter-efficient approach LoRA when lora_dim > 0)'''

import os
import datasets
import dp_transformers
import transformers
import sys
import logging
import torch
from dataclasses import dataclass, field
# from dp_transformers.layers.dp_merged_linear import mark_only_lora_as_trainable
# from dp_transformers.module_modification import convert_gpt2_attention_to_lora
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    train_file: str = field(default="./", metadata={
        "help": "Path to training data"
    })

    model_name: str = field(default="gpt2", metadata={
        "help": "Model name in HuggingFace, e.g. 'gpt2'"
    })

    lora_dim: int = field(default=0, metadata={
        "help": "LoRA dimension; 0 means LoRA is disabled"
    })

    sequence_len: int = field(default=128, metadata={
        "help": "Model sequence length"
    })

    lora_dropout: float = field(default=0.0, metadata={
        "help": "Dropout probability for LoRA layers"
    })

    lora_alpha: int = field(default=32, metadata={
        "help": "LoRA attention alpha"
    })


@dataclass
class Arguments:
    train: dp_transformers.TrainingArguments
    model: ModelArguments


def main(args: Arguments):
    transformers.set_seed(args.train.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = train_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {train_args.local_rank}, device: {train_args.device}, n_gpu: {train_args.n_gpu}, "
        f"distributed training: {bool(train_args.local_rank != -1)}, 16-bits training: {train_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {train_args}")

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model.model_name, attn_implementation='flash_attention_2' , torch_dtype=torch.bfloat16)
    model = model.to(train_args.device)

    dataset = datasets.load_dataset('json', data_files={'train': args.model.train_file})

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model.model_name)
    num_added_toks = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data
    input_embeddings_average = input_embeddings[:-num_added_toks].mean(dim=0, keepdim=True)
    output_embeddings_average = output_embeddings[:-num_added_toks].mean(dim=0, keepdim=True)

    input_embeddings[-num_added_toks:] = input_embeddings_average
    output_embeddings[-num_added_toks:] = output_embeddings_average


    label_column_names = [name for name in dataset["train"].column_names if "label" in name]
    label_map = { 0: "reject",  1: "granted", 2: "uncertain" }
    # Tokenize data
    def preprocess_function(examples):
        batch = []
        for t in range(len(examples['text'])):
            text = "\t".join(examples[label_column_names[0]][t]) + "\n\n" + examples['text'][t] + tokenizer.eos_token
            batch.append(text)

        result = tokenizer(batch, padding="longest", truncation=True,
                            max_length=args.model.sequence_len)
        return result
    
    train_data = dataset['train']
    with train_args.main_process_first(desc="tokenizing dataset"):
        train_data = train_data.map(
            preprocess_function, batched=True, desc="tokenizing dataset", remove_columns=dataset.column_names['train']
        )
    # if args.model.lora_dim > 0:
    #     model = convert_gpt2_attention_to_lora(
    #         model, r=args.model.lora_dim, lora_alpha=args.model.lora_alpha, lora_dropout=args.model.lora_dropout,
    #         enable_lora=[True, False, True], merge_weights=False
    #     )
    #     mark_only_lora_as_trainable(model)

    if train_args.local_rank == 0:
        logger.info(f"Total number of parameters of the model: {model.num_parameters(only_trainable=False)}")
        logger.info(f"Fine-tuned number of parameters of the model: {model.num_parameters(only_trainable=True)}")

    model = model.cuda()
    model.train()

    data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(tokenizer)

    train_dataloader = DataLoader(train_data, batch_size=6, shuffle=True, collate_fn=data_collator)

     # print only for rank 0
    if train_args.local_rank == 0:
        for batch in train_dataloader:
            print("*"*100)
            print("sample input")
            print(tokenizer.decode(batch['input_ids'][0]))
            res = []
            for i in range(len(batch['labels'][0])):
                if batch['labels'][0][i] != -100:
                    res.append(batch['labels'][0][i])
            print("sample output")
            print(tokenizer.decode(res))
            print("*"*100)
            break

    trainer = transformers.Trainer(
        args=train_args,
        model=model,
        train_dataset=train_data,
        #eval_dataset=train_data['test'],
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    train_result = trainer.train()

    if train_args.local_rank == 0 or train_args.local_rank == -1:
        metrics = train_result.metrics
        model.save_pretrained(args.train.output_dir + '/final')
        tokenizer.save_pretrained(args.train.output_dir + '/final')
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser((dp_transformers.TrainingArguments, ModelArguments))
    train_args, model_args = arg_parser.parse_args_into_dataclasses()
    main(Arguments(train=train_args, model=model_args))