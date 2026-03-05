import subprocess
import time
import sys
import re

def parse_speed(output):
    m = re.search(
        r"\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]",
        output
    )

    if m:
        prompt_tps = float(m.group(1))
        gen_tps = float(m.group(2))
        return (prompt_tps, gen_tps)
    else:
        print("Timing line not found.")
        return None

def parse_memory(output):
    for line in output.splitlines():
        if "- ROCm0" in line:
            nums = list(map(int, re.findall(r"\d+", line)))
            total, free, self_mem, model, context, compute, unaccounted = nums[2:9]
            return {
                "total": total,
                "free": free,
                "model": model,
                "context": context,
                "compute": compute,
                "unaccounted": unaccounted,
            }

def parse_token_count(output):

    prompt_tokens = None
    gen_tokens = None
    total_tokens = None
    
    m = re.search(r"prompt eval time.*?/\s*(\d+)\s*tokens", output)
    if m:
        prompt_tokens = int(m.group(1))
    else:
        print("Failed to parse for prompt token count...")

    m = re.search(r"eval time.*?/\s*(\d+)\s*tokens", output)
    if m:
        gen_tokens = int(m.group(1))
    else:
        print("Failed to parse for gen token count...")

    m = re.search(r"total time.*?/\s*(\d+)\s*tokens", output)
    if m:
        total_tokens = int(m.group(1))
    else:
        print("Failed to parse for total token count...")

    return prompt_tokens, gen_tokens, total_tokens

def parse_tokens_and_speed(output):
    pattern = re.compile(
r"""
prompt\ eval\ time\s*=\s*
([\d.]+)\s*ms\s*/\s*
(\d+)\s*tokens
\s*\(\s*
([\d.]+)\s*ms\ per\ token,\s*
([\d.]+)\s*tokens\ per\ second\)

\s*\n\s*

eval\ time\s*=\s*
([\d.]+)\s*ms\s*/\s*
(\d+)\s*tokens
\s*\(\s*
([\d.]+)\s*ms\ per\ token,\s*
([\d.]+)\s*tokens\ per\ second\)

\s*\n\s*

total\ time\s*=\s*
([\d.]+)\s*ms\s*/\s*
(\d+)\s*tokens
""",
re.VERBOSE
)

    m = pattern.search(output)

    if m:
        (
        prompt_ms,
        prompt_tokens,
        prompt_ms_per_tok,
        prompt_tps,

        eval_ms,
        eval_tokens,
        eval_ms_per_tok,
        eval_tps,

        total_ms,
        total_tokens
        ) = m.groups()

    prompt_ms = float(prompt_ms)
    prompt_tokens = int(prompt_tokens)
    prompt_tps = float(prompt_tps)

    eval_ms = float(eval_ms)
    eval_tokens = int(eval_tokens)
    eval_tps = float(eval_tps)

    total_ms = float(total_ms)
    total_tokens = int(total_tokens)
    
    return prompt_tokens, prompt_tps, eval_tokens, eval_tps, total_ms, total_tokens

def main():

    cmd = [
        "../llama/llama.cpp/build/bin/llama-cli",
        "-m", "../llama/llama.cpp/models/mistral.gguf",
        "-p", "Give a brief explanation of photosynthesis (1-3 lines)",
        "-n", "256",
        "--single-turn",
        "--simple-io",
        "--no-jinja",
        "-lv", "3",
      ]

    start = time.perf_counter()
    result = subprocess.run(cmd, text=True, capture_output=True)
    end = time.perf_counter()

    if result.returncode != 0:
        print("llama.cpp fail...")
        sys.exit(1)

    llama_output = result.stdout + result.stderr
    memory_output = result.stderr

    #prompt_tps, gen_tps = parse_speed(llama_output)
    memory_data = parse_memory(llama_output)
    #prompt_tokens, gen_tokens, total_tokens = parse_token_count(llama_output)
    prompt_tokens, prompt_tps, eval_tokens, eval_tps, total_ms, total_tokens = parse_tokens_and_speed(llama_output)

    print(f"Benchmark time {end-start} seconds")
    print(f"Prefill speed: {prompt_tps} tok/s")
    print(f"Decode speed: {eval_tps} tok/s")
    print(f"Prompt: {prompt_tokens} tokens")
    print(f"Generated: {eval_tokens} tokens")
    print(f"Inference time {total_ms} ms")
    print(f"Total: {total_tokens} tokens")
    print(f"total VRAM: {memory_data['total']} MiB")
    print(f"free VRAM: {memory_data['free']} MiB")
    print(f"used VRAM: {memory_data['total']-memory_data['free']} MiB")
    print(f"model VRAM: {memory_data['model']} MiB")
    print(f"context VRAM: {memory_data['context']} MiB")
    print(f"compute VRAM: {memory_data['compute']} MiB")
    print(f"unaccounted VRAM: {memory_data['unaccounted']} MiB")

if __name__ == "__main__":
    main()
