#!/usr/bin/env python3

import argparse
import logging
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


LOG = logging.getLogger("wmt26")

MODEL_PATH = "/model"

LANGUAGES = {
    "ces-deu": ("Czech", "German"),
    "cs-de": ("Czech", "German"),
    "cs-deu": ("Czech", "German"),
    "ces-de": ("Czech", "German"),
}

PROMPT = (
    "Translate the following text from {src} to {tgt}: {text}"
)


def str2bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lang-pair",
        required=True,
        choices=LANGUAGES,
    )
    parser.add_argument(
        "--batch-size",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--progress",
        action="store_true",
    )
    parser.add_argument(
        "--enforce-eager",
        type=str2bool,
        default=True,
        metavar="BOOL",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    return args


def make_prompt(lang_pair: str, text: str) -> str:
    src, tgt = LANGUAGES[lang_pair]
    return PROMPT.format(src=src, tgt=tgt, text=text)


def clean_output(text: str) -> str:
    return (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", " ")
        .strip()
    )


def read_batches(path: Path, batch_size: int):
    batch: list[str] = []

    with path.open("r", encoding="utf-8", errors="replace") as source:
        for raw_line in source:
            # Remove only the physical line terminator.
            line = raw_line.rstrip("\r\n")
            batch.append(line)

            if len(batch) == batch_size:
                yield batch
                batch = []

    if batch:
        yield batch


def translate_batch(
    llm: LLM,
    tokenizer,
    lang_pair: str,
    lines: list[str],
    progress: bool,
) -> list[str]:
    translations = [""] * len(lines)

    nonblank_positions = [
        index
        for index, line in enumerate(lines)
        if line.strip()
    ]

    if not nonblank_positions:
        return translations

    conversations = [
        [
            {
                "role": "user",
                "content": make_prompt(lang_pair, lines[index]),
            }
        ]
        for index in nonblank_positions
    ]

    prompt_token_ids = tokenizer.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    #print("PTI:", prompt_token_ids)
    input_ids = prompt_token_ids["input_ids"]

    prompts = [
        {"prompt_token_ids": token_ids}
        for token_ids in input_ids
    ]

    # Cap generation at 2x the longest prompt in the batch to reduce the risk of a long hallucination
    
    max_tokens = 2 * max(len(token_ids) for token_ids in input_ids)

    sampling_params = SamplingParams(
        temperature=0.0, # greedy
        max_tokens=max_tokens,
        frequency_penalty=0.1, # not good for translation with tags
        seed=0,
        logprobs=0,
    )


    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=progress,
    )
    #print(outputs)

    if len(outputs) != len(nonblank_positions):
        raise RuntimeError(
            f"vLLM returned {len(outputs)} outputs for "
            f"{len(nonblank_positions)} prompts"
        )

    for line_index, output in zip(
        nonblank_positions,
        outputs,
        strict=True,
    ):
        if not output.outputs:
            raise RuntimeError(
                f"vLLM returned no translation for input line {line_index + 1}"
            )

        translations[line_index] = clean_output(
            output.outputs[0].text
        )

    return translations


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=True,
        trust_remote_code=True
    )

    llm = LLM(
        model=MODEL_PATH,

        # Fixed deployment: one evaluator GPU.
        tensor_parallel_size=1,

        # Let the checkpoint select FP16/BF16 unless the final model
        # requires a specific dtype.
        dtype="bfloat16",
    
        # Fixed translation context limit.
        max_model_len=args.max_model_len,
        
        # dont pre-compile cuda graph; this is good for short benches
        enforce_eager=args.enforce_eager,

        # Keep scheduler concurrency aligned with evaluator batch size.
        max_num_seqs=args.batch_size,

        # Use this OR kv_cache_memory_bytes, not both.
        gpu_memory_utilization=0.90,

        # Change to "auto" if the pinned vLLM image does not support it.
        #kv_cache_dtype="turboquant_4bit_nc",

        tokenizer_mode="auto",
        tokenizer=MODEL_PATH,
        trust_remote_code=True,
        
        
        enable_prefix_caching=False,
        disable_log_stats=True,

        # Prevent generation_config.json from changing benchmark defaults.
        generation_config="vllm",
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    line_count = 0

    # Time only the translation work (tokenization + generation + write),
    # excluding model loading above.
    LOG.info("Translation started")
    start = time.perf_counter()

    with args.output.open("w", encoding="utf-8") as destination:
        for batch in read_batches(args.input, args.batch_size):
            translations = translate_batch(
                llm=llm,
                tokenizer=tokenizer,
                lang_pair=args.lang_pair,
                lines=batch,
                progress=args.progress,
            )

            for translation in translations:
                destination.write(translation + "\n")
                line_count += 1

            # Make each completed batch visible immediately.
            destination.flush()

    elapsed = time.perf_counter() - start

    LOG.info("wrote %d translations", line_count)
    LOG.info("total translation time: %.3f s", elapsed)
    LOG.info("batch size: %d", args.batch_size)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        LOG.exception("inference failed")
        raise SystemExit(1)

