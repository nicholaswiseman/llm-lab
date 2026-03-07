from bench import run_benchmark
import csv

DEFAULT_MODEL = "../llama/llama.cpp/models/mistral.gguf"
DEFAULT_PROMPT = "Give a brief explanation of photosynthesis (1-3 lines)"
DEFAULT_TOKENS = 256
DEFAULT_RUNS = 3

CONTEXT_VALUES = [1024, 2048, 4096, 8192]

def summarize(results):

    summary = []

    for block in results:

        context = block["context"]
        runs = block["results"]

        n = len(runs)

        decode = sum(r["decode_speed_tps"] for r in runs) / n
        prefill = sum(r["prefill_speed_tps"] for r in runs) / n
        infer = sum(r["inference_time_ms"] for r in runs) / n
        vram = sum(r["vram_used_mib"] for r in runs) / n

        summary.append({
            "context": context,
            "runs": n,
            "decode_tps_avg": round(decode, 2),
            "prefill_tps_avg": round(prefill, 2),
            "inference_ms_avg": round(infer, 2),
            "vram_used_mib_avg": round(vram, 2),
        })

    for row in summary:
        print(row)    

    return summary

def write_csv(all_results):
    #TODO: should add timestamp to csv filename to avoid overwriting data
    filename = "experiment.csv"
    fieldnames = [
        "context",
        "run",
        "benchmark_time_sec",
        "prefill_speed_tps",
        "decode_speed_tps",
        "prompt_tokens",
        "generated_tokens",
        "total_tokens",
        "inference_time_ms",
        "vram_total_mib",
        "vram_free_mib",
        "vram_used_mib",
        "vram_model_mib",
        "vram_context_mib",
        "vram_compute_mib",
        "vram_unaccounted_mib",
        "response"
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for block in all_results:
            context = block["context"]

            for i, run in enumerate(block["results"], start=1):

                row = {
                    "context": context,
                    "run": i,
                    **run
                }

                writer.writerow(row)

def main():

    all_results = []

    for context in CONTEXT_VALUES:
        print(f"Running context sweep for -c {context}")

        results = run_benchmark(
            model=DEFAULT_MODEL,
            prompt=DEFAULT_PROMPT,
            tokens=DEFAULT_TOKENS,
            context=context,
            runs=DEFAULT_RUNS,
        )

        all_results.append({
            "context": context,
            "results": results,
        })

        print(f"Finished context {context}")
        print("**********")

    print("Sweep complete")
    summarize(all_results)
    write_csv(all_results)

if __name__ == "__main__":
    main()
