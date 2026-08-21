import torch
from tqdm import tqdm
import transformers
import logging
from dataclasses import dataclass, field
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM

from datasets import load_dataset
from vllm import LLM, SamplingParams
import json

logger = logging.getLogger(__name__)


def batch(iterable, batch_size=16):
    """
    Yield successive batch_size-sized chunks from iterable.

    Args:
        iterable: Any iterable object (list, tuple, etc.)
        batch_size: Size of each batch (default: 16)

    Returns:
        Generator yielding lists of length batch_size (last batch may be shorter)

    Example:
        >>> list(batch([1,2,3,4,5], batch_size=2))
        [[1,2], [3,4], [5]]
    """
    iterator = iter(iterable)
    while True:
        batch_items = []
        try:
            for _ in range(batch_size):
                batch_items.append(next(iterator))
            yield batch_items
        except StopIteration:
            if batch_items:
                yield batch_items
            break


@dataclass
class ScriptArgs:
    original_train_file: str = field(default=None, metadata={"help": "Path to the train file"})

    is_multilabel: bool = field(default=False, metadata={"help": "Whether the dataset is multi-label"})

    checkpoint_file: str = field(default=None, metadata={"help": "Model name in HuggingFace, e.g. 'gpt2'"})
    max_sequence_len: int = field(default=128, metadata={"help": "Maximum sequence length to generate"})

    use_torch_load: bool = field(
        default=False,
        metadata={"help": "If your code sucks like mine and you save state dicts instead of models, use this flag."},
    )

    model_class: str = field(
        default="LlamaForCausalLM",
        metadata={"help": "If your code sucks like mine and you save state dicts instead of models, use this flag."},
    )

    model_string: str = field(
        default="meta-llama/Llama-3.2-1B",
        metadata={"help": "If your code sucks like mine and you save state dicts instead of models, use this flag."},
    )

    debug: bool = field(default=False, metadata={"help": "If true just generate 5 samples and dont save them."})
    dataset: str = field(default=None)
    output_file: str = field(default=None)
    dataset_description: str = field(default="")
    temperature: float = field(default=0.6)
    batch_size: int = field(default=16)


def main(script_args: ScriptArgs):
    assert (script_args.output_file and script_args.original_train_file) or (
        script_args.dataset
    ), "Need either explicit output file name or dataset & dataset_description args to create file"
    output_file = (
        script_args.output_file
        or f"data/{script_args.dataset}/dp-transformers/train{script_args.dataset_description}-dp-transformers.jsonl"
    )
    original_train_file = (
        script_args.original_train_file
        or f"data/{script_args.dataset}/original/train{script_args.dataset_description}-original.jsonl"
    )
    print("temperature", script_args.temperature)
    sampling_params = SamplingParams(max_tokens=script_args.max_sequence_len, temperature=script_args.temperature)
    llm = LLM(model=script_args.checkpoint_file, tensor_parallel_size=1, gpu_memory_utilization=0.7)
    # prepare prompts
    prompts = []
    ds = load_dataset("json", data_files=original_train_file)["train"]
    if script_args.debug:
        ds = ds.select([0, 1, 2, 3, 4])
    for d in tqdm(ds, desc="Preparing prompts..."):
        label = "\t".join(d["label"]).strip()
        prompts.append(f"{label}\n\n")

    if script_args.batch_size <=0:
        raise ValueError
    outputs = []
    with open(output_file, "w") as f:    
        for start in range(0, len(prompts), script_args.batch_size):
            end = min(len(prompts), start + script_args.batch_size)
            input_prompts = prompts[start:end]

            results = llm.generate(input_prompts, sampling_params)
            batch_ds = ds.select(range(start,end))
            for d, output in zip(batch_ds, results):
                prompt = output.prompt
                generated_text = output.outputs[0].text
                # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
                result = {
                    "prompt": prompt,
                    "text": generated_text,
                    "label": d["label"]
                }
                f.write(json.dumps(result) + "\n")
                outputs.append(result)
    
    if len(outputs) != len(prompts):
        raise ValueError
    # if script_args.debug:
    #     for o in outputs:
    #         print(f"Prompt: {o['prompt']!r}, Generated text: {o['text']!r}")
        
    
    # with open(output_file, "w") as f:
    #     for o in outputs:



if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser((ScriptArgs,))
    script_args, *_ = arg_parser.parse_args_into_dataclasses()
    main(script_args)
