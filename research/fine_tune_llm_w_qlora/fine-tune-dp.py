# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Train LLMs with DP using QLoRA'''

import datasets
import dp_transformers
import transformers
import sys
import logging
import torch
import ast
import linear
import data_utils
import copy
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Union
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

from pynvml import *
from torch.utils.data import DataLoader

# def print_gpu_utilization():
#     nvmlInit()
#     handle = nvmlDeviceGetHandleByIndex(0)
#     info = nvmlDeviceGetMemoryInfo(handle)
#     print(f"GPU memory occupied: {info.used//1024**2} MB.")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name: str = field(default="gpt2", metadata={
        "help": "Model name in HuggingFace, e.g. 'gpt2'"
    })
    dataset_name: str = field(default="sst2", metadata={
        "help": "Dataset name in HuggingFace, e.g. 'sst2'"
    })
    sequence_len: int = field(default=128, metadata={
        "help": "Maximum sequence length"
    })

@dataclass
class ScriptArgs:
    train_file: str = field(default=None, metadata={
        "help": "Path to the train file"
    })
@dataclass
class LoraArguments:
    enable_lora: bool = field(default=False, metadata={
        "help": "Whether to enable LoRA"
    })
    lora_dim: int = field(default=8, metadata={
        "help": "LoRA dimension"
    })
    lora_alpha: int = field(default=8, metadata={
        "help": "LoRA alpha"
    })
    lora_dropout: float = field(default=0.0, metadata={
        "help": "LoRA dropout"
    })

    target_modules: List[str] = field(
        default_factory=list,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )

    def as_peft_config(self) -> LoraConfig:
        if not self.enable_lora:
            raise ValueError("LoRA is not enabled, cannot convert to LoRA config")
        params = asdict(self)
        params.pop("enable_lora")
        params["r"] = params.pop("lora_dim")
        params["target_modules"] = ast.literal_eval(params["target_modules"][0])
        return LoraConfig(**params)


@dataclass
class Arguments:
    train: dp_transformers.TrainingArguments
    privacy: dp_transformers.PrivacyArguments
    script_args: ScriptArgs
    model: ModelArguments
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

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model.model_name)
    num_added_toks = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print("adding special tokens " , num_added_toks)
    
    # Load dataset
    dataset = datasets.load_dataset('json', data_files={'train': args.script_args.train_file})
    print("dataset", dataset)

    def preprocess_function(examples):
        batch = []
        for t in range(len(examples['text'])):
            text = "\t".join(examples['label'][t]) + "\n\n" + examples['text'][t] + tokenizer.eos_token
            batch.append(text)
        
        result = tokenizer(batch, truncation=True, padding="longest",  max_length=args.model.sequence_len)

        return result


    bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model.model_name, quantization_config=bnb_config)
    model.enable_input_require_grads()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=train_args.gradient_checkpointing)

    
    model.resize_token_embeddings(len(tokenizer))


    train_data = dataset['train'].map(preprocess_function, batched=True, desc="tokenizing dataset", batch_size=16, remove_columns=dataset.column_names['train'])
    if args.lora.enable_lora:
        logger.info("Using LoRA")
        model = get_peft_model(model=model, peft_config=args.lora.as_peft_config())
    else:
        logger.info("Not using LoRA")

    if train_args.local_rank == 0:
        logger.info(f"Total number of parameters of the model: {model.num_parameters(only_trainable=False)}")
        logger.info(f"Fine-tuned number of parameters of the model: {model.num_parameters(only_trainable=True)}")
    
    data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(tokenizer)
    train_dataloader = DataLoader(train_data, batch_size=train_args.per_device_train_batch_size, shuffle=True, collate_fn=data_collator)

    print("train data post processing", train_data)
    if train_args.local_rank == 0:
        for batch in train_dataloader:
            print("*"*100)
            print("sample input")
            print(tokenizer.decode(batch['input_ids'][0]))
            res = []
            for i in range(len(batch['labels'][0])):
                if batch['labels'][0][i] != -100:
                    res.append(batch['labels'][0][i])
            print("last token of the sample output", batch['labels'][0][-1])
            print("sample output")
            print(tokenizer.decode(res))
            print("*"*100)
            print("attention mask", batch['attention_mask'][0])
            break
    

    trainer = dp_transformers.dp_utils.OpacusDPTrainer(
        args=train_args,
        model=model,
        data_collator=data_collator,
        train_dataset=train_data,
        tokenizer=tokenizer,
        privacy_args=privacy_args,
    )

    if hasattr(trainer.model._module, "config"):
        # The following is for GradSampleModule wrapping
        ignore_keys = getattr(trainer.model._module.config, "keys_to_ignore_at_inference", [])
    elif hasattr(trainer.model._module.module, "config"):
        # The following is for GradSampleModule and DPDDP wrapping
        ignore_keys = getattr(trainer.model._module.module.config, "keys_to_ignore_at_inference", [])
    else:
        ignore_keys = []

    try:
        # A workaround to avoid the following error:
        # AttributeError: 'GradSampleModule' object has no attribute 'gradient_checkpointing_enable'
        # inside Trainer _inner_training_loop. Already done by prepare_model_for_kbit_training
        trainer.args.gradient_checkpointing = False
        result = trainer.train(ignore_keys_for_eval=ignore_keys)
    finally:
        eps_prv = trainer.get_prv_epsilon()
        eps_rdp = trainer.get_rdp_epsilon()
        trainer.log({
            "final_epsilon_prv": eps_prv,
            "final_epsilon_rdp": eps_rdp
        })

    # if dataset.run_test:
    #     logger.info("Running test set evaluation after training")   
    #     test_metrics = dataset.compute_test_metrics(trainer)
    #     trainer.log(test_metrics)

    # def print_summary(result):
    #     print(f"Time: {result.metrics['train_runtime']:.2f}")
    #     print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    #     print_gpu_utilization()

    # print_summary(result)

if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser((dp_transformers.TrainingArguments, dp_transformers.PrivacyArguments, ScriptArgs, ModelArguments, LoraArguments))
    train_args, privacy_args, script_args, model_args, lora_args = arg_parser.parse_args_into_dataclasses()
    main(Arguments(train=train_args, privacy=privacy_args, script_args=script_args, model=model_args, lora=lora_args))
