"""
Velaris Motion - Performance & Stability Benchmarking Suite
Target Release: v1.1.0
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from config import config
from security import SafeGenerationRequest
from engine import HybridEngine
from job_queue import Job, JobStatus


def load_benchmark_prompts(count: int = 10) -> List[str]:
    prompts_file = Path(__file__).resolve().parent / "benchmark_prompts.json"
    if not prompts_file.exists():
        raise FileNotFoundError(f"Benchmark prompts file '{prompts_file}' not found.")
    
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    if count > len(prompts):
        count = len(prompts)
    return prompts[:count]


class BenchmarkRunner:
    def __init__(self, mock_mode: bool = False):
        self.mock_mode = mock_mode
        self.engine = HybridEngine(mock_mode=mock_mode)

    def run(self, prompt_count: int = 10, seed_start: int = 42) -> Dict[str, Any]:
        prompts = load_benchmark_prompts(prompt_count)
        print(f"\n==================================================================")
        print(f" VELARIS MOTION BENCHMARK: Running {len(prompts)} prompts (Mock={self.mock_mode})")
        print(f"==================================================================\n")

        results = []
        latencies = []
        peak_vrams = []
        success_count = 0
        oom_count = 0
        other_fail_count = 0

        for i, prompt in enumerate(prompts):
            seed = seed_start + i
            req = SafeGenerationRequest(
                prompt=prompt,
                seed=seed,
                num_inference_steps=25,
                dream_steps=30,
                fps=7,
                motion_bucket_id=127,
            )
            job = Job(req)
            
            print(f"[{i+1}/{len(prompts)}] Prompt: \"{prompt[:50]}...\" (Seed: {seed})")
            t0 = time.time()
            is_oom = False
            status_str = "SUCCESS"
            error_msg = None

            try:
                self.engine.execute_job(job)
                elapsed = time.time() - t0
                success_count += 1
                latencies.append(elapsed)
                
                vram_peak = job.metadata.get("vram_peak_mb", 0.0) if job.metadata else 0.0
                peak_vrams.append(vram_peak)
                print(f"      -> SUCCESS in {elapsed:.2f}s (Peak VRAM: {vram_peak:.1f} MB)")
            except Exception as e:
                elapsed = time.time() - t0
                error_msg = str(e)
                if "out of memory" in error_msg.lower() or "cuda oom" in error_msg.lower():
                    oom_count += 1
                    is_oom = True
                    status_str = "OOM_FAILURE"
                    print(f"      -> CUDA OOM ERROR in {elapsed:.2f}s")
                else:
                    other_fail_count += 1
                    status_str = "FAILED"
                    print(f"      -> FAILED in {elapsed:.2f}s: {error_msg}")

            results.append({
                "index": i + 1,
                "prompt": prompt,
                "seed": seed,
                "status": status_str,
                "duration_seconds": round(elapsed, 2),
                "is_oom": is_oom,
                "error": error_msg,
            })

        total = len(prompts)
        success_rate = (success_count / total) * 100.0 if total > 0 else 0.0
        oom_rate = (oom_count / total) * 100.0 if total > 0 else 0.0

        median_latency = float(np.median(latencies)) if latencies else 0.0
        p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0
        min_latency = float(np.min(latencies)) if latencies else 0.0
        max_latency = float(np.max(latencies)) if latencies else 0.0
        max_vram = float(np.max(peak_vrams)) if peak_vrams else 0.0

        report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "velaris_version": config.version,
            "mock_mode": self.mock_mode,
            "device": self.engine.device,
            "total_prompts": total,
            "success_count": success_count,
            "success_rate_percent": round(success_rate, 2),
            "oom_count": oom_count,
            "oom_rate_percent": round(oom_rate, 2),
            "other_failures": other_fail_count,
            "latency": {
                "median_seconds": round(median_latency, 2),
                "p95_seconds": round(p95_latency, 2),
                "min_seconds": round(min_latency, 2),
                "max_seconds": round(max_latency, 2),
            },
            "peak_vram_mb": round(max_vram, 2),
            "resolution": "1024x576",
            "video_length_frames": 14,
            "fps": 7,
            "detailed_results": results,
        }

        return report


def format_markdown_report(report: Dict[str, Any]) -> str:
    md = []
    md.append(f"# Velaris Motion v{report['velaris_version']} Benchmark Report\n")
    md.append(f"**Date:** {report['timestamp']}  ")
    md.append(f"**Device:** {report['device']} | **Mock Mode:** {report['mock_mode']}  ")
    md.append(f"**Resolution:** {report['resolution']} | **Frames:** {report['video_length_frames']} @ {report['fps']} fps\n")
    
    md.append("## Summary Metrics\n")
    md.append("| Metric | Value |")
    md.append("| :--- | :--- |")
    md.append(f"| **Total Prompts Evaluated** | `{report['total_prompts']}` |")
    md.append(f"| **Generation Success Rate** | `{report['success_rate_percent']}%` ({report['success_count']}/{report['total_prompts']}) |")
    md.append(f"| **Out-of-Memory (OOM) Rate** | `{report['oom_rate_percent']}%` ({report['oom_count']}/{report['total_prompts']}) |")
    md.append(f"| **Median Latency** | `{report['latency']['median_seconds']}s` |")
    md.append(f"| **P95 Latency** | `{report['latency']['p95_seconds']}s` |")
    md.append(f"| **Min Latency** | `{report['latency']['min_seconds']}s` |")
    md.append(f"| **Max Latency** | `{report['latency']['max_seconds']}s` |")
    md.append(f"| **Overall Peak VRAM** | `{report['peak_vram_mb']} MB` |\n")

    md.append("## Per-Prompt Execution Log\n")
    md.append("| # | Prompt Preview | Status | Duration (s) |")
    md.append("| :--- | :--- | :--- | :--- |")
    for r in report["detailed_results"]:
        prompt_snippet = r['prompt'][:45] + "..." if len(r['prompt']) > 45 else r['prompt']
        status_badge = "✅ PASS" if r['status'] == "SUCCESS" else f"❌ {r['status']}"
        md.append(f"| {r['index']} | {prompt_snippet} | {status_badge} | {r['duration_seconds']}s |")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Velaris Motion Pipeline Benchmark")
    parser.add_argument("--prompts", type=int, choices=[10, 20, 50], default=10, help="Number of benchmark prompts")
    parser.add_argument("--mock", action="store_true", help="Run with mock pipeline for fast validation")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output path for benchmark report")
    args = parser.parse_args()

    runner = BenchmarkRunner(mock_mode=args.mock)
    report = runner.run(prompt_count=args.prompts)

    # Save JSON report
    output_path = Path(args.output).resolve()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save Markdown report
    md_path = output_path.with_suffix(".md")
    md_content = format_markdown_report(report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n==================================================================")
    print(f" BENCHMARK COMPLETE")
    print(f" Success Rate: {report['success_rate_percent']}% | Median Latency: {report['latency']['median_seconds']}s")
    print(f" JSON Saved to: {output_path}")
    print(f" Markdown Saved to: {md_path}")
    print(f"==================================================================\n")


if __name__ == "__main__":
    main()
