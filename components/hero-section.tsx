"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    Sparkles,
    ArrowRight,
    Play,
    XCircle,
    CheckCircle2,
    Cpu,
    Database,
    Clock,
    RefreshCw,
    Image as ImageIcon,
    UploadCloud,
    Film,
    SlidersHorizontal,
    Trash2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { AdvancedControls, DEFAULT_PARAMS, GenerationParams } from "./advanced-controls";
import { GenerationHistory, HistoryItem } from "./generation-history";

type GenerationStage =
    | "idle"
    | "queued"
    | "loading_model"
    | "generating_image"
    | "loading_video_model"
    | "animating"
    | "exporting"
    | "completed"
    | "failed"
    | "cancelled";

interface JobMetadata {
    seed?: number;
    generation_duration_seconds?: number;
    vram_peak_mb?: number;
    resolution?: { width: number; height: number };
    models?: Record<string, { model_id: string; revision: string }>;
    fps?: number;
}

interface JobSnapshot {
    job_id: string;
    status: string;
    stage: GenerationStage;
    prompt: string;
    progress: number;
    message: string;
    video_url?: string;
    duration_seconds?: number;
    error?: string;
    metadata?: JobMetadata;
    is_img2vid?: boolean;
}

interface SceneConcept {
    id: string;
    title: string;
    category: string;
    badge: string;
    enhanced_prompt: string;
    negative_prompt: string;
    camera_movement: string;
    lighting: string;
}

const STAGE_LABELS: Record<GenerationStage, string> = {
    idle: "Ready to Dream",
    queued: "Queued in GPU Queue",
    loading_model: "Loading Dreamer (SD 1.5) into VRAM",
    generating_image: "Synthesizing Base Frame",
    loading_video_model: "Unloading Dreamer & Loading SVD-XT Animator",
    animating: "Synthesizing Motion Frames with SVD-XT",
    exporting: "Encoding & Exporting Video",
    completed: "Generation Complete",
    failed: "Generation Failed",
    cancelled: "Generation Cancelled",
};

const VAULT_STORAGE_KEY = "velaris_generation_vault_v2";

export function HeroSection() {
    const [mode, setMode] = useState<"text" | "image">("text");
    const [prompt, setPrompt] = useState("");
    const [params, setParams] = useState<GenerationParams>(DEFAULT_PARAMS);
    const [selectedImage, setSelectedImage] = useState<File | null>(null);
    const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
    const [isDragging, setIsDragging] = useState(false);

    const [currentJobId, setCurrentJobId] = useState<string | null>(null);
    const [jobState, setJobState] = useState<JobSnapshot | null>(null);
    const [isGenerating, setIsGenerating] = useState(false);
    const [isCancelling, setIsCancelling] = useState(false);
    const [backendStatus, setBackendStatus] = useState<string>("Checking Engine...");
    const [error, setError] = useState<string | null>(null);
    const [history, setHistory] = useState<HistoryItem[]>([]);
    const [ideationConcepts, setIdeationConcepts] = useState<SceneConcept[]>([]);
    const [isIdeating, setIsIdeating] = useState(false);

    const fileInputRef = useRef<HTMLInputElement | null>(null);
    const eventSourceRef = useRef<EventSource | null>(null);

    const API_BASE = "http://localhost:8000";

    const handleIdeate = async () => {
        setIsIdeating(true);
        setError(null);
        try {
            const queryPrompt = prompt.trim() || "a magnificent golden eagle";
            const res = await fetch(`${API_BASE}/ideate?prompt=${encodeURIComponent(queryPrompt)}`);
            if (!res.ok) throw new Error("Fikir servisine erişilemedi.");
            const data = await res.json();
            setIdeationConcepts(data.concepts || []);
        } catch (err: any) {
            console.error("Ideation error:", err);
            setError("Fikir üretimi sırasında bir hata oluştu.");
        } finally {
            setIsIdeating(false);
        }
    };

    const handleSelectConcept = (concept: SceneConcept) => {
        setPrompt(concept.enhanced_prompt);
        setParams((prev) => ({
            ...prev,
            negative_prompt: concept.negative_prompt,
            engine: "studio",
        }));
        setIdeationConcepts([]);
    };

    // Load history from localStorage on mount
    useEffect(() => {
        try {
            const raw = localStorage.getItem(VAULT_STORAGE_KEY);
            if (raw) {
                setHistory(JSON.parse(raw));
            }
        } catch (e) {
            console.error("Failed to load history vault from localStorage:", e);
        }
    }, []);

    // Check backend health on mount
    useEffect(() => {
        fetch(`${API_BASE}/health`)
            .then((res) => res.json())
            .then((data) => {
                const dev = data.vram?.device_name || (data.device === "cuda" ? "RTX 4060 GPU" : "CPU");
                setBackendStatus(`Velaris v${data.version} | ${dev} Ready`);
            })
            .catch(() => {
                setBackendStatus("Backend Offline (Run run_velaris.bat)");
            });
    }, []);

    const saveToVault = (item: HistoryItem) => {
        setHistory((prev) => {
            const filtered = prev.filter((h) => h.id !== item.id);
            const updated = [item, ...filtered].slice(0, 30);
            try {
                localStorage.setItem(VAULT_STORAGE_KEY, JSON.stringify(updated));
            } catch (err) {
                console.warn("localStorage quota exceeded:", err);
            }
            return updated;
        });
    };

    const deleteFromVault = (id: string) => {
        setHistory((prev) => {
            const updated = prev.filter((item) => item.id !== id);
            localStorage.setItem(VAULT_STORAGE_KEY, JSON.stringify(updated));
            return updated;
        });
    };

    const clearVault = () => {
        setHistory([]);
        localStorage.removeItem(VAULT_STORAGE_KEY);
    };

    const handleReuse = (item: HistoryItem) => {
        setPrompt(item.prompt);
        if (item.seed !== undefined) {
            setParams((p) => ({
                ...p,
                seed: item.seed ?? null,
                motion_bucket_id: item.motionBucketId ?? p.motion_bucket_id,
                num_inference_steps: item.numInferenceSteps ?? p.num_inference_steps,
                audio_mode: item.audioMode ?? p.audio_mode,
            }));
        }
        window.scrollTo({ top: 0, behavior: "smooth" });
    };

    // Stop SSE stream
    const stopStreaming = () => {
        if (eventSourceRef.current) {
            eventSourceRef.current.close();
            eventSourceRef.current = null;
        }
    };

    // Start SSE stream (Sprint 3A)
    const startStreaming = (jobId: string, currentMode: "text" | "image") => {
        stopStreaming();
        const es = new EventSource(`${API_BASE}/jobs/${jobId}/stream`);
        eventSourceRef.current = es;

        es.onmessage = (event) => {
            try {
                const data: JobSnapshot = JSON.parse(event.data);
                setJobState(data);
                if (
                    data.status === "COMPLETED" ||
                    data.status === "FAILED" ||
                    data.status === "CANCELLED"
                ) {
                    setIsGenerating(false);
                    setIsCancelling(false);
                    if (data.status === "FAILED") {
                        setError(data.error || "Generation encountered an error.");
                    } else if (data.status === "COMPLETED" && data.video_url) {
                        saveToVault({
                            id: data.job_id,
                            prompt: data.prompt,
                            videoUrl: data.video_url,
                            createdAt: Date.now(),
                            durationSeconds: data.duration_seconds || data.metadata?.generation_duration_seconds,
                            vramPeakMb: data.metadata?.vram_peak_mb,
                            fps: params.interpolate ? 24 : params.fps,
                            seed: data.metadata?.seed,
                            audioMode: params.audio_mode,
                            isImg2Vid: currentMode === "image",
                            motionBucketId: params.motion_bucket_id,
                            numInferenceSteps: params.num_inference_steps,
                        });
                    }
                    es.close();
                }
            } catch (e) {
                console.error("SSE parse error:", e);
            }
        };

        es.onerror = () => {
            console.warn("SSE connection dropped — stream closed.");
            es.close();
        };
    };

    useEffect(() => {
        return () => stopStreaming();
    }, []);

    // Handle Image file selection
    const handleFileChange = (file: File | null) => {
        if (!file) return;
        const validTypes = ["image/png", "image/jpeg", "image/webp"];
        if (!validTypes.includes(file.type)) {
            setError("Invalid image format. Please select a PNG, JPG, or WEBP image.");
            return;
        }
        if (file.size > 15 * 1024 * 1024) {
            setError("File size exceeds 15MB limit.");
            return;
        }
        setError(null);
        setSelectedImage(file);
        setImagePreviewUrl(URL.createObjectURL(file));
        if (!prompt) {
            setPrompt(`Cinematic motion animation of ${file.name.replace(/\.[^/.]+$/, "")}`);
        }
    };

    const handleClearImage = () => {
        setSelectedImage(null);
        if (imagePreviewUrl) {
            URL.revokeObjectURL(imagePreviewUrl);
            setImagePreviewUrl(null);
        }
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    };

    const handleGenerate = async () => {
        if (isGenerating) return;

        if (mode === "text" && !prompt.trim()) {
            setError("Please enter a prompt to generate video.");
            return;
        }

        if (mode === "image" && !selectedImage) {
            setError("Please select or drop an image to animate.");
            return;
        }

        setIsGenerating(true);
        setError(null);
        setJobState(null);
        setIsCancelling(false);

        try {
            let response: Response;

            if (mode === "text") {
                // Text-to-Video Mode
                response = await fetch(`${API_BASE}/generate`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        prompt: prompt.trim(),
                        ...params,
                        ...(params.seed === null ? {} : { seed: params.seed }),
                    }),
                });
            } else {
                // Sprint 4A: Direct Image-to-Video Mode
                const formData = new FormData();
                formData.append("image", selectedImage as File);
                formData.append("motion_bucket_id", String(params.motion_bucket_id));
                formData.append("num_inference_steps", String(params.num_inference_steps));
                formData.append("fps", String(params.fps));
                if (params.seed !== null) {
                    formData.append("seed", String(params.seed));
                }
                formData.append("interpolate", String(params.interpolate));
                formData.append("audio_mode", params.audio_mode);
                formData.append("quality", params.quality);
                formData.append("duration_mode", params.duration_mode);
                if (params.target_duration != null) {
                    formData.append("target_duration", String(params.target_duration));
                }
                if (prompt.trim()) {
                    formData.append("prompt", prompt.trim());
                }
                formData.append("aspect_ratio", params.aspect_ratio);
                formData.append("eco_mode", String(params.eco_mode));

                response = await fetch(`${API_BASE}/animate-image`, {
                    method: "POST",
                    body: formData,
                });
            }

            const data = await response.json();

            if (response.status !== 202) {
                const errMsg = data.error?.message || data.detail || "Failed to submit job";
                throw new Error(errMsg);
            }

            setCurrentJobId(data.job_id);
            setJobState(data);
            startStreaming(data.job_id, mode);
        } catch (err: any) {
            setError(err.message || "Failed to connect to Velaris Engine. Is it running on port 8000?");
            setIsGenerating(false);
        }
    };

    const handleCancel = async () => {
        if (!currentJobId || isCancelling) return;
        setIsCancelling(true);
        try {
            await fetch(`${API_BASE}/jobs/${currentJobId}/cancel`, { method: "POST" });
        } catch (err: any) {
            console.error("Cancellation error:", err);
        }
    };

    const currentStage: GenerationStage = jobState?.stage || "idle";
    const stageLabel = STAGE_LABELS[currentStage] || "Processing";
    const progressPercent = Math.round((jobState?.progress || 0) * 100);

    return (
        <section className="relative min-h-screen flex flex-col items-center justify-start overflow-hidden pt-28 pb-20">
            {/* Background Atmosphere */}
            <div className="absolute inset-0 z-0 pointer-events-none">
                <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[850px] h-[550px] bg-purple-600/15 rounded-full blur-[130px] opacity-40 animate-blob" />
                <div className="absolute top-1/4 left-1/3 w-[600px] h-[400px] bg-indigo-600/10 rounded-full blur-[110px] opacity-30 animate-blob animation-delay-2000" />
            </div>

            <div className="relative z-10 w-full max-w-5xl px-6 flex flex-col items-center text-center space-y-7">
                {/* Engine Health Badge */}
                <motion.div
                    initial={{ opacity: 0, y: 15 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                    className="space-y-3"
                >
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-white/10 bg-white/5 text-xs font-mono text-white/80 backdrop-blur-sm">
                        <span className="relative flex h-2 w-2">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                        </span>
                        {backendStatus}
                    </div>

                    <h1 className="text-5xl md:text-7xl font-bold tracking-tight text-white">
                        Imagine it. <br />
                        <span className="bg-clip-text text-transparent bg-gradient-to-r from-purple-400 via-indigo-300 to-cyan-400">
                            Watch it come alive.
                        </span>
                    </h1>

                    <p className="max-w-xl mx-auto text-base text-white/60 leading-relaxed font-sans">
                        Production video synthesis studio. Sequential VRAM management, 24fps frame interpolation, and AI soundscapes.
                    </p>
                </motion.div>

                {/* Mode Selector Tabs (Sprint 4A) */}
                <div className="flex items-center p-1 rounded-2xl bg-white/[0.04] border border-white/10 backdrop-blur-md">
                    <button
                        onClick={() => { setMode("text"); setError(null); }}
                        className={cn(
                            "flex items-center gap-2 px-5 py-2 rounded-xl text-xs font-semibold transition-all cursor-pointer",
                            mode === "text"
                                ? "bg-white text-black shadow-lg"
                                : "text-white/60 hover:text-white"
                        )}
                    >
                        <Sparkles className="w-4 h-4" />
                        Text to Video
                    </button>

                    <button
                        onClick={() => { setMode("image"); setError(null); }}
                        className={cn(
                            "flex items-center gap-2 px-5 py-2 rounded-xl text-xs font-semibold transition-all cursor-pointer",
                            mode === "image"
                                ? "bg-white text-black shadow-lg"
                                : "text-white/60 hover:text-white"
                        )}
                    >
                        <ImageIcon className="w-4 h-4" />
                        Image to Video
                        <span className="text-[10px] px-1.5 py-0.2 rounded bg-purple-500/20 text-purple-700 font-mono">
                            ~45% faster
                        </span>
                    </button>
                </div>

                {/* Input Card Container */}
                <motion.div
                    layout
                    initial={{ opacity: 0, y: 15 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4 }}
                    className="w-full max-w-2xl relative group text-left"
                >
                    <div className="absolute -inset-0.5 bg-gradient-to-r from-purple-500/40 via-indigo-500/30 to-blue-500/40 rounded-3xl blur opacity-30 group-hover:opacity-50 transition duration-1000"></div>

                    <div className="relative bg-[#09090b]/90 border border-white/10 rounded-3xl p-5 shadow-2xl backdrop-blur-xl flex flex-col gap-4">
                        {mode === "text" ? (
                            /* Text-to-Video Input */
                            <div className="space-y-3">
                                <div className="flex items-center gap-3">
                                    <div className="p-2 rounded-xl bg-purple-500/10 text-purple-400">
                                        <Sparkles className="w-5 h-5" />
                                    </div>
                                    <input
                                        type="text"
                                        value={prompt}
                                        onChange={(e) => setPrompt(e.target.value)}
                                        placeholder="Describe a cinematic scene (e.g., 'A cyberpunk street in neon rain...')"
                                        className="flex-1 bg-transparent text-white placeholder:text-white/30 text-sm focus:outline-none"
                                        onKeyDown={(e) => e.key === "Enter" && !isGenerating && handleGenerate()}
                                        disabled={isGenerating}
                                        id="prompt-input"
                                    />
                                    {/* ── Fikir Üret Button ── */}
                                    <button
                                        type="button"
                                        onClick={handleIdeate}
                                        disabled={isGenerating || isIdeating}
                                        title="AI ile 3 sinematik sahne konsepti üret"
                                        className={cn(
                                            "flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold transition-all shrink-0 border",
                                            isIdeating
                                                ? "border-purple-500/40 bg-purple-500/10 text-purple-400 cursor-wait"
                                                : "border-purple-500/30 bg-purple-500/10 text-purple-300 hover:bg-purple-500/20 hover:text-purple-200 cursor-pointer",
                                            (isGenerating || isIdeating) && "opacity-50"
                                        )}
                                    >
                                        {isIdeating
                                            ? <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                                            : <Sparkles className="w-3.5 h-3.5" />
                                        }
                                        Fikir Üret
                                    </button>
                                </div>

                                {/* ── Ideation Concept Cards ── */}
                                <AnimatePresence>
                                    {ideationConcepts.length > 0 && (
                                        <motion.div
                                            key="ideation-panel"
                                            initial={{ opacity: 0, y: -6 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            exit={{ opacity: 0, y: -6 }}
                                            transition={{ duration: 0.25 }}
                                            className="flex flex-col gap-2 pt-1"
                                        >
                                            <p className="text-[10px] font-mono text-purple-400/70 uppercase tracking-widest pl-1">
                                                ✦ Sahne Konseptleri — birini seçin
                                            </p>
                                            {ideationConcepts.map((concept) => (
                                                <button
                                                    key={concept.id}
                                                    type="button"
                                                    onClick={() => handleSelectConcept(concept)}
                                                    className="group w-full text-left px-3.5 py-3 rounded-xl border border-white/8 bg-white/[0.025] hover:border-purple-500/40 hover:bg-purple-500/[0.07] transition-all"
                                                >
                                                    <div className="flex items-start justify-between gap-2">
                                                        <div className="flex-1 min-w-0">
                                                            <div className="flex items-center gap-2 mb-1">
                                                                <span className="text-[10px] font-bold font-mono px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-300 uppercase tracking-wide">
                                                                    {concept.badge}
                                                                </span>
                                                                <span className="text-xs font-semibold text-white/85 truncate">
                                                                    {concept.title}
                                                                </span>
                                                            </div>
                                                            <p className="text-[11px] text-white/45 line-clamp-2 leading-relaxed">
                                                                {concept.enhanced_prompt}
                                                            </p>
                                                        </div>
                                                        <span className="text-[10px] text-white/30 shrink-0 mt-0.5 group-hover:text-purple-400 transition-colors">
                                                            📷 {concept.camera_movement}
                                                        </span>
                                                    </div>
                                                </button>
                                            ))}
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </div>
                        ) : (
                            /* Sprint 4A: Direct Image-to-Video Drag & Drop Upload Zone */
                            <div className="space-y-4">
                                <input
                                    type="file"
                                    ref={fileInputRef}
                                    accept="image/png,image/jpeg,image/webp"
                                    className="hidden"
                                    onChange={(e) => handleFileChange(e.target.files?.[0] || null)}
                                />

                                {!imagePreviewUrl ? (
                                    <div
                                        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                                        onDragLeave={() => setIsDragging(false)}
                                        onDrop={(e) => {
                                            e.preventDefault();
                                            setIsDragging(false);
                                            handleFileChange(e.dataTransfer.files?.[0] || null);
                                        }}
                                        onClick={() => fileInputRef.current?.click()}
                                        className={cn(
                                            "border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-all flex flex-col items-center justify-center gap-2",
                                            isDragging
                                                ? "border-purple-400 bg-purple-500/10"
                                                : "border-white/10 hover:border-white/20 bg-white/[0.02]"
                                        )}
                                    >
                                        <div className="p-3 rounded-2xl bg-white/5 border border-white/10 text-white/70">
                                            <UploadCloud className="w-6 h-6" />
                                        </div>
                                        <p className="text-xs font-semibold text-white/90">
                                            Drag &amp; drop an image, or <span className="text-purple-400 underline">browse</span>
                                        </p>
                                        <p className="text-[11px] text-white/40">
                                            PNG, JPG, or WEBP up to 15MB • Dreamer is skipped for instant motion
                                        </p>
                                    </div>
                                ) : (
                                    <div className="flex items-center gap-4 p-3 rounded-2xl bg-white/[0.03] border border-white/10">
                                        <img
                                            src={imagePreviewUrl}
                                            alt="Upload preview"
                                            className="w-20 h-14 object-cover rounded-xl border border-white/10"
                                        />
                                        <div className="flex-1 min-w-0">
                                            <p className="text-xs font-semibold text-white truncate">
                                                {selectedImage?.name}
                                            </p>
                                            <p className="text-[10px] text-white/40 font-mono mt-0.5">
                                                {(Number(selectedImage?.size || 0) / 1024 / 1024).toFixed(2)} MB • Ready to animate
                                            </p>
                                        </div>
                                        <button
                                            type="button"
                                            onClick={handleClearImage}
                                            disabled={isGenerating}
                                            className="p-2 rounded-xl text-white/40 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                                            title="Remove image"
                                        >
                                            <Trash2 className="w-4 h-4" />
                                        </button>
                                    </div>
                                )}

                                {/* Optional guidance prompt for Image-to-Video */}
                                <div className="flex items-center gap-2 pt-1 border-t border-white/5">
                                    <Film className="w-4 h-4 text-white/40 shrink-0 ml-1" />
                                    <input
                                        type="text"
                                        value={prompt}
                                        onChange={(e) => setPrompt(e.target.value)}
                                        placeholder="Optional motion prompt (e.g. 'gentle camera pan with flying leaves')"
                                        className="flex-1 bg-transparent text-white placeholder:text-white/30 text-xs focus:outline-none"
                                        disabled={isGenerating}
                                    />
                                </div>
                            </div>
                        )}

                        {/* Action buttons bar */}
                        <div className="flex items-center justify-between pt-3 border-t border-white/5">
                            <span className="text-[11px] font-mono text-white/40">
                                {mode === "text" ? "SD 1.5 + SVD-XT" : "SVD-XT Direct Mode (No Dreamer)"}
                            </span>

                            {isGenerating ? (
                                <button
                                    id="cancel-btn"
                                    onClick={handleCancel}
                                    disabled={isCancelling}
                                    className="flex items-center gap-2 bg-red-500/20 hover:bg-red-500/30 text-red-300 border border-red-500/30 px-5 py-2.5 rounded-xl font-medium text-xs transition-all cursor-pointer"
                                >
                                    <XCircle className="w-4 h-4" />
                                    {isCancelling ? "Cancelling..." : "Cancel"}
                                </button>
                            ) : (
                                <button
                                    id="generate-btn"
                                    onClick={handleGenerate}
                                    disabled={mode === "text" ? !prompt.trim() : !selectedImage}
                                    className={cn(
                                        "flex items-center gap-2 bg-white text-black px-6 py-2.5 rounded-xl font-semibold text-xs transition-all shadow-lg cursor-pointer",
                                        (mode === "text" ? !prompt.trim() : !selectedImage)
                                            ? "opacity-50 cursor-not-allowed"
                                            : "hover:scale-[1.02] active:scale-[0.98]"
                                    )}
                                >
                                    {mode === "text" ? "Dream & Animate" : "Animate Image"}
                                    <ArrowRight className="w-4 h-4 ml-0.5" />
                                </button>
                            )}
                        </div>
                    </div>
                </motion.div>

                {/* Advanced Generation Controls Panel (Sprint 3B) */}
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4, delay: 0.15 }}
                    className="w-full max-w-2xl"
                >
                    <AdvancedControls
                        params={params}
                        onChange={setParams}
                        disabled={isGenerating}
                    />
                </motion.div>

                {/* Error Banner */}
                {error && (
                    <motion.div
                        initial={{ opacity: 0, y: -5 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="text-red-400 bg-red-500/10 px-4 py-2.5 rounded-xl border border-red-500/20 max-w-xl text-xs flex items-center gap-2"
                    >
                        <XCircle className="w-4 h-4 shrink-0" />
                        <span>{error}</span>
                    </motion.div>
                )}

                {/* Real-Time Live Progress Indicator (Sprint 3A via SSE) */}
                <AnimatePresence>
                    {isGenerating && jobState && (
                        <motion.div
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            className="w-full max-w-2xl bg-white/[0.04] border border-white/10 rounded-2xl p-5 shadow-2xl backdrop-blur-md text-left space-y-3"
                        >
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                    <div className="w-4 h-4 border-2 border-purple-400 border-t-transparent rounded-full animate-spin" />
                                    <span className="text-sm font-semibold text-white/90">{stageLabel}</span>
                                </div>
                                <span className="text-xs font-mono text-purple-300 bg-purple-500/10 px-2 py-0.5 rounded border border-purple-500/20">
                                    {progressPercent}%
                                </span>
                            </div>

                            {/* Deterministic progress bar driven by backend SSE push */}
                            <div className="w-full bg-white/10 h-2 rounded-full overflow-hidden">
                                <motion.div
                                    className="bg-gradient-to-r from-purple-500 via-indigo-500 to-cyan-400 h-full"
                                    initial={{ width: 0 }}
                                    animate={{ width: `${progressPercent}%` }}
                                    transition={{ duration: 0.3 }}
                                />
                            </div>

                            <p className="text-xs text-white/50 truncate font-mono">
                                {jobState.message || "Communicating with Velaris GPU worker..."}
                            </p>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Completed Video Player */}
                {jobState && jobState.status === "COMPLETED" && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.96 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="w-full max-w-3xl mt-4 rounded-3xl overflow-hidden border border-white/10 bg-white/[0.03] shadow-2xl backdrop-blur-md"
                    >
                        {jobState.video_url && (
                            <video
                                src={`${API_BASE}${jobState.video_url}`}
                                controls
                                autoPlay
                                loop
                                className="w-full h-auto aspect-video object-cover"
                            />
                        )}

                        {/* Reproducibility Metadata Bar */}
                        {jobState.metadata && (
                            <div className="p-4 bg-black/50 border-t border-white/10 text-left grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                                <div className="space-y-1">
                                    <span className="text-white/40 flex items-center gap-1 font-mono text-[11px]">
                                        <Database className="w-3 h-3" /> Seed
                                    </span>
                                    <span className="font-mono text-white/80">{jobState.metadata.seed ?? "N/A"}</span>
                                </div>
                                <div className="space-y-1">
                                    <span className="text-white/40 flex items-center gap-1 font-mono text-[11px]">
                                        <Clock className="w-3 h-3" /> Duration
                                    </span>
                                    <span className="font-mono text-white/80">
                                        {jobState.metadata.generation_duration_seconds ? `${jobState.metadata.generation_duration_seconds}s` : "N/A"}
                                    </span>
                                </div>
                                <div className="space-y-1">
                                    <span className="text-white/40 flex items-center gap-1 font-mono text-[11px]">
                                        <Cpu className="w-3 h-3" /> Peak VRAM
                                    </span>
                                    <span className="font-mono text-white/80">
                                        {jobState.metadata.vram_peak_mb ? `${Math.round(jobState.metadata.vram_peak_mb)} MB` : "N/A"}
                                    </span>
                                </div>
                                <div className="space-y-1">
                                    <span className="text-white/40 flex items-center gap-1 font-mono text-[11px]">
                                        <CheckCircle2 className="w-3 h-3 text-emerald-400" /> Output
                                    </span>
                                    <span className="font-mono text-white/80">
                                        1024×576 • {params.interpolate ? "24fps (RIFE)" : `${params.fps}fps`}
                                    </span>
                                </div>
                            </div>
                        )}
                    </motion.div>
                )}

                {/* Sample Prompts (when idle) */}
                {!jobState && mode === "text" && (
                    <motion.div
                        initial={{ opacity: 0, y: 30 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6, delay: 0.3 }}
                        className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-6 w-full max-w-4xl"
                    >
                        {[
                            "A cyberpunk street in neon rain, reflections on wet asphalt",
                            "Calm tropical beach with crystal turquoise water gently lapping",
                            "An astronaut walking on an alien surface under two glowing moons"
                        ].map((sample, i) => (
                            <div
                                key={i}
                                onClick={() => setPrompt(sample)}
                                className="group relative rounded-2xl overflow-hidden bg-white/[0.02] border border-white/5 p-4 cursor-pointer hover:border-purple-500/40 hover:bg-white/[0.04] transition-all text-left"
                            >
                                <div className="text-xs text-white/70 line-clamp-2">
                                    "{sample}"
                                </div>
                            </div>
                        ))}
                    </motion.div>
                )}
            </div>

            {/* Sprint 5A: Generation Vault / History Gallery */}
            <GenerationHistory
                items={history}
                onReuse={handleReuse}
                onDelete={deleteFromVault}
                onClearAll={clearVault}
            />
        </section>
    );
}
