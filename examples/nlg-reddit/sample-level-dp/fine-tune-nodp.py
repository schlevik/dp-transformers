# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Train GPT2 model series without DP (w/ parameter-efficient approach LoRA when lora_dim > 0)'''

import datasets
import dp_transformers
import transformers
import sys
import logging
import torch
from dataclasses import dataclass, field
from dataclasses import dataclass, field, asdict
from peft import get_peft_model, LoraConfig
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
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
    lora_alpha: int = field(default=8, metadata={
        "help": "LoRA alpha"
    })
    lora_dropout: float = field(default=0.0, metadata={
        "help": "LoRA dropout"
    })

    def as_peft_config(self) -> LoraConfig:
        if not self.enable_lora:
            raise ValueError("LoRA is not enabled, cannot convert to LoRA config")
        params = asdict(self)
        params.pop("enable_lora")
        params["r"] = params.pop("lora_dim")
        return LoraConfig(**params)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels, attention_mask = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "attention_mask"))
        return dict(
            input_ids= torch.tensor(input_ids),
            labels= torch.tensor(labels),
            attention_mask= torch.tensor(attention_mask),
        )
@dataclass
class Arguments:
    train: dp_transformers.TrainingArguments
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

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(args.script_args.model_name)
    model = model.to(train_args.device)

    # Load data
    dataset = datasets.load_dataset('json', data_files=script_args.train_file)

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.script_args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize data
    def preprocess_function(examples):
        text = "\t".join([item for item in examples['label']])+ "\n\n" + examples['text'] + tokenizer.eos_token    
        result = tokenizer(text, truncation=True, padding="max_length", padding_side="left", max_length=args.script_args.sequence_len)
        result['labels'] = result['input_ids'].copy()
        for i in range(len(result['attention_mask'])):
            if result['attention_mask'][i] == 0:
                result['labels'][i] = -100
        return result

    # Tokenize data
    with train_args.main_process_first(desc="tokenizing dataset"):
        # dataset = dataset.map(
        #     lambda batch: tokenizer(batch['text'], padding="max_length", truncation=True, max_length=args.script_args.sequence_len),
        #     batched=True, num_proc=8, desc="tokenizing dataset", remove_columns=dataset.column_names['train']
        # )
        final_data = dataset['train']
        final_data = final_data.map(preprocess_function, desc="tokenizing dataset",  remove_columns=dataset.column_names['train']).train_test_split(test_size=0.1)


    print("final data is " , final_data)
    print("sample data is " , tokenizer.decode(final_data['train']['input_ids'][0]))
    print("sample data label" , tokenizer.decode([x for x in final_data['train']['labels'][0] if x != -100]))

    if args.lora.enable_lora:
        logger.info("Using LoRA")
        model = get_peft_model(model=model, peft_config=args.lora.as_peft_config())
    else:
        logger.info("Not using LoRA")

    if train_args.local_rank == 0:
        logger.info(f"Total number of parameters of the model: {model.num_parameters(only_trainable=False)}")
        logger.info(f"Fine-tuned number of parameters of the model: {model.num_parameters(only_trainable=True)}")

    model = model.cuda()
    model.train()

    data_collator = DataCollatorForSupervisedDataset(tokenizer) #dp_transformers.DataCollatorForPrivateCausalLanguageModeling(tokenizer)
    dataloader = DataLoader(final_data['train'], batch_size=16, shuffle=True, collate_fn=data_collator)
    
    trainer = transformers.Trainer(
        args=train_args,
        model=model,
        train_dataset=final_data['train'],
        eval_dataset=final_data['test'],
        data_collator=data_collator,
    )

    trainer.train()
    if args.lora.enable_lora:
        print("Merging model...")
        model = model.merge_and_unload()
    model.save_pretrained(args.train.output_dir)
    tokenizer.save_pretrained(args.train.output_dir)

if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser((dp_transformers.TrainingArguments, ScriptArgs, LoraArguments))
    train_args, script_args, lora_args = arg_parser.parse_args_into_dataclasses()
    main(Arguments(train=train_args, script_args=script_args, lora=lora_args))
