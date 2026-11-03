"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
    Activity,
    ArrowLeft,
    CheckCircle2,
    Clock,
    Cpu,
    Gauge,
    HardDrive,
    Play,
    RefreshCw,
    Sparkles,
    XCircle,
    Zap,
} from "lucide-react";
import { SiteHeader } from "@/components/site-header";

interface LatencyMetrics {
    median_seconds: number;
    p95_seconds: number;
    min_seconds: number;
    max_seconds: number;
}

interface BenchmarkDetail {
    index: number;
    prompt: string;
    seed: number;
    status: string;
    duration_seconds: number;
    is_oom: boolean;
    error: string | null;
}

interface BenchmarkReport {
    timestamp: string;
    velaris_version: string;
    mock_mode: boolean;
    device: string;
    total_prompts: number;
    success_count: number;
    success_rate_percent: number;
    oom_count: number;
    oom_rate_percent: number;
    other_failures: number;
    latency: LatencyMetrics;
    peak_vram_mb: number;
    resolution: string;
    video_length_frames: number;
    fps: number;
    detailed_results: BenchmarkDetail[];
}

export default function BenchmarkPage() {
    const [report, setReport] = useState<BenchmarkReport | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [isRunning, setIsRunning] = useState(false);
    const [promptCount, setPromptCount] = useState<number>(5);
    const [error, setError] = useState<string | null>(null);

    const API_BASE = "http://localhost:8000";

    const fetchBenchmarkResults = async () => {
        setIsLoading(true);
        setError(null);
        try {
            const res = await fetch(`${API_BASE}/benchmark-results`);
            if (!res.ok) throw new Error("Could not fetch benchmark data");
            const data: BenchmarkReport = await res.json();
            setReport(data);
        } catch (err: any) {
            setError(err.message || "Failed to load benchmark results. Make sure backend is running.");
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchBenchmarkResults();
    }, []);

    const handleRunBenchmark = async () => {
        setIsRunning(true);
        setError(null);
        try {
            const res = await fetch(`${API_BASE}/benchmark/run?mock=true&prompts=${promptCount}`, {
                method: "POST",
            });
            if (!res.ok) throw new Error("Benchmark run failed");
            const data: BenchmarkReport = await res.json();
            setReport(data);
        } catch (err: any) {
            setError(err.message || "Failed to execute benchmark run.");
        } finally {
            setIsRunning(false);
        }
    };

    return (
        <div className="min-h-screen bg-black text-white selection:bg-purple-500/20">
            <SiteHeader />

            <main className="max-w-6xl mx-auto px-6 pt-28 pb-20">
                {/* Header navigation & title */}
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-10 pb-6 border-b border-white/10">
                    <div>
                        <Link
                            href="/"
                            className="inline-flex items-center gap-2 text-xs font-medium text-white/50 hover:text-white mb-3 transition-colors"
                        >
                            <ArrowLeft className="w-3.5 h-3.5" />
                            Back to Studio
                        </Link>
                        <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
                            <span className="p-2 rounded-xl bg-purple-500/20 border border-purple-500/30 text-purple-400">
                                <Gauge className="w-6 h-6" />
                            </span>
                            Velaris Benchmark Lab
                        </h1>
                        <p className="text-sm text-white/50 mt-1">
                            Real-time latency, peak VRAM containment, and throughput verification under synthetic load.
                        </p>
                    </div>

                    <div className="flex items-center gap-3">
                        <select
                            value={promptCount}
                            onChange={(e) => setPromptCount(Number(e.target.value))}
                            disabled={isRunning}
                            aria-label="Benchmark prompt count"
                            className="bg-white/5 border border-white/10 text-white text-xs rounded-xl px-3 py-2 outline-none focus:border-purple-500"
                        >
                            <option value={3} className="bg-zinc-900">3 Prompts (Fast)</option>
                            <option value={5} className="bg-zinc-900">5 Prompts (Standard)</option>
                            <option value={10} className="bg-zinc-900">10 Prompts (Full Suite)</option>
                        </select>

                        <button
                            onClick={handleRunBenchmark}
                            disabled={isRunning}
                            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white text-xs font-semibold shadow-lg shadow-purple-500/20 disabled:opacity-50 transition-all cursor-pointer"
                        >
                            {isRunning ? (
                                <>
                                    <RefreshCw className="w-4 h-4 animate-spin" />
                                    Running Suite...
                                </>
                            ) : (
                                <>
                                    <Play className="w-4 h-4 fill-white" />
                                    Run Benchmark
                                </>
                            )}
                        </button>
                    </div>
                </div>

                {error && (
                    <div className="mb-6 p-4 rounded-xl border border-red-500/30 bg-red-500/10 text-red-400 text-xs flex items-center gap-2">
                        <XCircle className="w-4 h-4 shrink-0" />
                        <span>{error}</span>
                    </div>
                )}

                {/* KPI Metrics Cards */}
                {report && (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                        <motion.div
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="p-5 rounded-2xl border border-white/10 bg-white/[0.03] backdrop-blur-md"
                        >
                            <div className="flex items-center justify-between text-white/50 text-xs mb-2">
                                <span>Success Rate</span>
                                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                            </div>
                            <div className="text-2xl font-bold font-mono text-emerald-400">
                                {report.success_rate_percent}%
                            </div>
                            <div className="text-[11px] text-white/40 mt-1">
                                {report.success_count} of {report.total_prompts} passed
                            </div>
                        </motion.div>

                        <motion.div
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.05 }}
                            className="p-5 rounded-2xl border border-white/10 bg-white/[0.03] backdrop-blur-md"
                        >
                            <div className="flex items-center justify-between text-white/50 text-xs mb-2">
                                <span>Median Latency</span>
                                <Clock className="w-4 h-4 text-cyan-400" />
                            </div>
                            <div className="text-2xl font-bold font-mono text-cyan-400">
                                {report.latency.median_seconds.toFixed(2)}s
                            </div>
                            <div className="text-[11px] text-white/40 mt-1">
                                P95: {report.latency.p95_seconds.toFixed(2)}s
                            </div>
                        </motion.div>

                        <motion.div
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.1 }}
                            className="p-5 rounded-2xl border border-white/10 bg-white/[0.03] backdrop-blur-md"
                        >
                            <div className="flex items-center justify-between text-white/50 text-xs mb-2">
                                <span>Peak VRAM</span>
                                <HardDrive className="w-4 h-4 text-purple-400" />
                            </div>
                            <div className="text-2xl font-bold font-mono text-purple-400">
                                {report.peak_vram_mb > 0 ? `${Math.round(report.peak_vram_mb)} MB` : "< 1 GB"}
                            </div>
                            <div className="text-[11px] text-white/40 mt-1">
                                RTX 4060 (8GB Target)
                            </div>
                        </motion.div>

                        <motion.div
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.15 }}
                            className="p-5 rounded-2xl border border-white/10 bg-white/[0.03] backdrop-blur-md"
                        >
                            <div className="flex items-center justify-between text-white/50 text-xs mb-2">
                                <span>OOM Occurrences</span>
                                <Zap className="w-4 h-4 text-amber-400" />
                            </div>
                            <div className="text-2xl font-bold font-mono text-white">
                                {report.oom_count}
                            </div>
                            <div className="text-[11px] text-white/40 mt-1">
                                VRAM Orchestrator Active
                            </div>
                        </motion.div>
                    </div>
                )}

                {/* System Specs Badge Banner */}
                {report && (
                    <div className="mb-8 p-4 rounded-2xl border border-white/10 bg-white/[0.02] flex flex-wrap items-center justify-between gap-4 text-xs font-mono text-white/60">
                        <div className="flex items-center gap-6">
                            <span>Velaris v{report.velaris_version}</span>
                            <span>Device: {report.device.toUpperCase()}</span>
                            <span>Resolution: {report.resolution}</span>
                            <span>Frames: {report.video_length_frames} @ {report.fps}fps</span>
                        </div>
                        <div className="text-white/40">
                            Last Run: {new Date(report.timestamp).toLocaleString()}
                        </div>
                    </div>
                )}

                {/* Execution Details Table */}
                <div className="rounded-2xl border border-white/10 bg-white/[0.02] overflow-hidden backdrop-blur-md">
                    <div className="px-6 py-4 border-b border-white/10 flex items-center justify-between">
                        <h2 className="text-sm font-semibold tracking-tight text-white flex items-center gap-2">
                            <Activity className="w-4 h-4 text-purple-400" />
                            Per-Prompt Execution Log
                        </h2>
                        <span className="text-xs text-white/40 font-mono">
                            {report?.detailed_results.length || 0} evaluated
                        </span>
                    </div>

                    {report && report.detailed_results.length > 0 ? (
                        <div className="overflow-x-auto">
                            <table className="w-full text-left text-xs">
                                <thead className="bg-white/[0.02] border-b border-white/5 text-white/40 font-mono">
                                    <tr>
                                        <th className="py-3 px-4 w-12">#</th>
                                        <th className="py-3 px-4">Prompt</th>
                                        <th className="py-3 px-4 w-28">Seed</th>
                                        <th className="py-3 px-4 w-28">Duration</th>
                                        <th className="py-3 px-4 w-32 text-right">Status</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-white/5 font-mono">
                                    {report.detailed_results.map((row) => (
                                        <tr key={row.index} className="hover:bg-white/[0.02] transition-colors">
                                            <td className="py-3 px-4 text-white/40">{row.index}</td>
                                            <td className="py-3 px-4 font-sans text-white/90 line-clamp-1 max-w-md">
                                                "{row.prompt}"
                                            </td>
                                            <td className="py-3 px-4 text-white/50">{row.seed}</td>
                                            <td className="py-3 px-4 text-white/70">{row.duration_seconds.toFixed(2)}s</td>
                                            <td className="py-3 px-4 text-right">
                                                {row.status === "SUCCESS" ? (
                                                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                                                        <CheckCircle2 className="w-3 h-3" />
                                                        PASS
                                                    </span>
                                                ) : (
                                                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-red-500/10 text-red-400 border border-red-500/20">
                                                        <XCircle className="w-3 h-3" />
                                                        {row.status}
                                                    </span>
                                                )}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    ) : (
                        <div className="p-12 text-center text-white/40 text-xs">
                            No benchmark logs available. Click &quot;Run Benchmark&quot; to evaluate system performance.
                        </div>
                    )}
                </div>
            </main>
        </div>
    );
}
