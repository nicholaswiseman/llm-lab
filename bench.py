import argparse
from runner import run_llama

DEFAULT_LLAMA = "../llama/llama.cpp/build/bin/llama-cli"
DEFAULT_MODEL = "../llama/llama.cpp/models/mistral.gguf"
DEFAULT_PROMPT = "Give a brief explanation of photosynthesis (1-3 lines)"
DEFAULT_TOKENS = 256
DEFAULT_CONTEXT = 4096
DEFAULT_RUNS = 1

def build_cmd(model, prompt, tokens, context):

    cmd = [
        DEFAULT_LLAMA,
        "-m", model,
        "-p", prompt,
        "-n", str(tokens),
        "-c", str(context),
        "--single-turn",
        "--simple-io",
        "--no-jinja",
        "-lv", "3",
      ]

    return cmd

def run_benchmark(model, prompt, tokens, context, runs):
    if model is None:
        model = DEFAULT_MODEL
    if prompt is None: 
        prompt = DEFAULT_PROMPT
    if tokens is None:
        tokens = DEFAULT_TOKENS
    if runs is None:
        runs = DEFAULT_RUNS

    cmd = build_cmd(model, prompt, tokens, context)
    results = []
    for i in range(runs):
        result = run_llama(cmd)
        results.append(result)
    return results

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT, help="context size")    
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="number of benchmark runs")

    args = parser.parse_args()
    result = run_benchmark(args.model, args.prompt, args.tokens, args.context, args.runs)
    print(result)

if __name__ == "__main__":
    main()
