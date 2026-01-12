"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Sparkles, ArrowRight, Play } from "lucide-react";
import { cn } from "@/lib/utils";

export function HeroSection() {
    const [prompt, setPrompt] = useState("");
    const [isGenerating, setIsGenerating] = useState(false);
    const [prediction, setPrediction] = useState<any>(null);
    const [error, setError] = useState<string | null>(null);

    const handleGenerate = async () => {
        if (!prompt) return;
        setIsGenerating(true);
        setError(null);
        setPrediction(null);

        try {
            // Call Local Velaris Engine
            const response = await fetch("http://localhost:8000/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ prompt }),
            });

            const data = await response.json();

            if (response.status !== 200) {
                throw new Error(data.detail || "Failed to generate video");
            }

            // Local engine maps "video_url" directly
            setPrediction({
                output: data.video_url,
                status: "succeeded"
            });

        } catch (err: any) {
            setError(err.message || "Something went wrong. Is the local backend running?");
        } finally {
            setIsGenerating(false);
        }
    };

    return (
        <section className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden pt-20 pb-20">
            {/* Background Elements */}
            <div className="absolute inset-0 z-0">
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[500px] bg-purple-500/20 rounded-full blur-[120px] opacity-30 animate-blob" />
                <div className="absolute top-1/4 left-1/3 w-[600px] h-[400px] bg-blue-500/10 rounded-full blur-[100px] opacity-20 animate-blob animation-delay-2000" />
            </div>

            <div className="relative z-10 w-full max-w-5xl px-6 flex flex-col items-center text-center space-y-8">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                    className="space-y-4"
                >
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-white/10 bg-white/5 text-xs font-medium text-white/80 backdrop-blur-sm">
                        <span className="relative flex h-2 w-2">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                        </span>
                        Generative Video AI Model 1.0
                    </div>

                    <h1 className="text-5xl md:text-7xl font-bold tracking-tighter leading-tight text-white mb-4">
                        Imagine it. <br />
                        <span className="text-gradient">Watch it come alive.</span>
                    </h1>

                    <p className="max-w-xl mx-auto text-lg text-white/60 leading-relaxed">
                        Transform your ideas into stunning high-definition videos with our state-of-the-art AI model. Just type a prompt and let the magic happen.
                    </p>
                </motion.div>

                {/* Prompt Input Box */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: 0.2 }}
                    className="w-full max-w-2xl relative group"
                >
                    <div className="absolute -inset-0.5 bg-gradient-to-r from-purple-500/50 to-blue-500/50 rounded-2xl blur opacity-30 group-hover:opacity-50 transition duration-1000"></div>
                    <div className="relative flex items-center bg-[#0A0A0A] border border-white/10 rounded-2xl p-2 shadow-2xl">
                        <div className="pl-4">
                            <Sparkles className="w-5 h-5 text-white/40" />
                        </div>
                        <input
                            type="text"
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            placeholder="Describe a cinematic scene, e.g., 'A cyberpunk city in rain...'"
                            className="flex-1 bg-transparent border-none text-white placeholder:text-white/30 px-4 py-3 focus:outline-none focus:ring-0 text-base"
                            onKeyDown={(e) => e.key === "Enter" && !isGenerating && handleGenerate()}
                        />
                        <button
                            onClick={handleGenerate}
                            disabled={!prompt || isGenerating}
                            className={cn(
                                "flex items-center gap-2 bg-white text-black px-6 py-3 rounded-xl font-medium transition-all",
                                !prompt ? "opacity-50 cursor-not-allowed" : "hover:scale-[1.02] active:scale-[0.98]",
                                isGenerating && "opacity-80 cursor-wait"
                            )}
                        >
                            {isGenerating ? "Generating..." : (
                                <>
                                    Generate
                                    <ArrowRight className="w-4 h-4 ml-1" />
                                </>
                            )}
                        </button>
                    </div>
                </motion.div>

                {/* Status / Error Message */}
                {error && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-red-400 bg-red-400/10 px-4 py-2 rounded-lg border border-red-400/20">
                        {error}
                    </motion.div>
                )}

                {/* Result Area */}
                {prediction && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="w-full max-w-3xl mt-8 rounded-2xl overflow-hidden border border-white/10 bg-white/5 shadow-2xl"
                    >
                        {prediction.output ? (
                            <video
                                src={prediction.output}
                                controls
                                autoPlay
                                loop
                                className="w-full h-auto"
                            />
                        ) : (
                            <div className="aspect-video flex flex-col items-center justify-center space-y-4">
                                <div className="w-10 h-10 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                                <p className="text-white/50 text-sm animate-pulse">
                                    {isGenerating ? "Velaris Engine is dreaming & animating..." : "Ready to dream"}
                                </p>
                            </div>
                        )}
                    </motion.div>
                )}

                {/* Sample Gallery (Push down or hide when generating) */}
                {!prediction && (
                    <motion.div
                        initial={{ opacity: 0, y: 40 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8, delay: 0.4 }}
                        className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-10 opacity-50 hover:opacity-100 transition-opacity"
                    >
                        {[1, 2, 3].map((i) => (
                            <div key={i} className="group relative aspect-video rounded-xl overflow-hidden bg-white/5 border border-white/5">
                                <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity bg-black/40">
                                    <Play className="w-10 h-10 text-white fill-white" />
                                </div>
                                <div className="w-full h-full bg-gradient-to-br from-white/5 to-white/0" />
                                <div className="absolute bottom-3 left-3 text-xs text-white/70 font-medium">Prompt Example #{i}</div>
                            </div>
                        ))}
                    </motion.div>
                )}
            </div>
        </section>
    );
}
