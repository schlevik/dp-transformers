# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Train GPT2 model series with DP (w/ parameter-efficient approach LoRA when lora_dim > 0)'''

import math
import datasets
import torch
import dp_transformers
import transformers
import sys
import logging

from dataclasses import dataclass, field, asdict
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from typing import List
import ast 
import wandb

logger = logging.getLogger(__name__)


@dataclass
class ScriptArgs:
    train_file: str = field(default=None, metadata={
        "help": "Path to the train file"
    })

    model_name: str = field(default="gpt2", metadata={
        "help": "Model name in HuggingFace, e.g. 'gpt2'"
    })
    sequence_len: int = field(default=128, metadata={
        "help": "Maximum sequence length"
    })


@dataclass
class LoraArguments:
    enable_lora: bool = field(default=False, metadata={
        "help": "Whether to enable LoRA"
    })
    lora_dim: int = field(default=8, metadata={
        "help": "LoRA dimension"
    })
    lora_alpha: int = field(default=16, metadata={
        "help": "LoRA alpha"
    })
    lora_dropout: float = field(default=0.0, metadata={
        "help": "LoRA dropout"
    })
    target_modules: List[str] = field(default_factory=list, metadata={
        "help": "List of module names or regex expression of the module names to replace with Lora."
        "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
    })

    def as_peft_config(self) -> LoraConfig:
        if not self.enable_lora:
            raise ValueError("LoRA is not enabled, cannot convert to LoRA config")
        params = asdict(self)
        params.pop("enable_lora")
        params["r"] = params.pop("lora_dim")
        params["target_modules"] = ast.literal_eval(params["target_modules"][0])
        params['use_rslora'] = True
        return LoraConfig(**params)


@dataclass
class Arguments:
    train: dp_transformers.TrainingArguments
    privacy: dp_transformers.PrivacyArguments
    script_args: ScriptArgs
    lora: LoraConfig



@dataclass
class Arguments:
    train: dp_transformers.TrainingArguments
    privacy: dp_transformers.PrivacyArguments
    script_args: ScriptArgs
    lora: LoraArguments


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
    logger.info(f"Privacy parameters {privacy_args}")

    if args.lora.enable_lora:
        print("Lora Training Enabled.....")
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(args.script_args.model_name, attn_implementation='flash_attention_2',  quantization_config=bnb_config)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=train_args.gradient_checkpointing)
        model = get_peft_model(model=model, peft_config=args.lora.as_peft_config())
    else:
    # Load model
        model = AutoModelForCausalLM.from_pretrained(args.script_args.model_name, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)
        model = model.to(train_args.device)

    dataset = datasets.load_dataset('json', data_files={'train': args.script_args.train_file})

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.script_args.model_name)
    num_added_toks = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print("adding special tokens " , num_added_toks)
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
                            max_length=args.script_args.sequence_len)
        return result
    
    train_data = dataset['train']
    with train_args.main_process_first(desc="tokenizing dataset"):
        train_data = train_data.map(
            preprocess_function, batched=True, desc="tokenizing dataset", remove_columns=dataset.column_names['train']
        )

    print("padding token" , tokenizer.pad_token)

    logger.info(f"Total number of parameters of the model: {model.num_parameters(only_trainable=False)}")
    logger.info(f"Number of trainable parameters of the model: {model.num_parameters(only_trainable=True)}")

    num_samples = len(train_data)
    privacy_args.target_delta = 1.0/(num_samples**2)
    
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
    
    print("training arguments", train_args)
    trainer = dp_transformers.dp_utils.OpacusDPTrainer(
        args=train_args,
        model=model,
        train_dataset=train_data,
        # eval_dataset=train_data['test'],
        data_collator=data_collator,
        privacy_args=privacy_args,
        tokenizer=tokenizer,
    )

    try:
        train_result = trainer.train()
    finally:
        eps_prv = trainer.get_prv_epsilon()
        eps_rdp = trainer.get_rdp_epsilon()
        trainer.log({
            "final_epsilon_prv": eps_prv,
            "final_epsilon_rdp": eps_rdp
        })

    if train_args.local_rank == 0 or train_args.local_rank == -1:
        if args.lora.enable_lora:
            model.save_pretrained(args.train.output_dir + '/final_lora')
            tokenizer.save_pretrained(args.train.output_dir + '/final_lora')
            del model
            model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(args.script_args.model_name), args.train.output_dir + '/final_lora')
            model = model.merge_and_unload()
           
        metrics = train_result.metrics
        # trainer.save_model()
        model.save_pretrained(args.train.output_dir + '/final')
        tokenizer.save_pretrained(args.train.output_dir + '/final')
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)


if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser((dp_transformers.TrainingArguments, dp_transformers.PrivacyArguments, ScriptArgs, LoraArguments))
    train_args, privacy_args, script_args, lora_args = arg_parser.parse_args_into_dataclasses()
    main(Arguments(train=train_args, privacy=privacy_args, script_args=script_args, lora=lora_args))