"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    SlidersHorizontal,
    Footprints,
    Gauge,
    Dices,
    Lock,
    Unlock,
    Music,
    Mic,
    VolumeX,
    Zap,
    ChevronDown,
    Film,
    Maximize2,
    Clock,
    Cpu,
    Sparkles,
    Leaf,
    Smartphone,
    Monitor,
    Square,
    Video,
} from "lucide-react";
import { cn } from "@/lib/utils";

export interface GenerationParams {
    dream_steps: number;
    num_inference_steps: number;
    motion_bucket_id: number;
    fps: number;
    guidance_scale: number;
    seed: number | null;
    interpolate: boolean;
    audio_mode: "none" | "music" | "narration" | "foley";
    negative_prompt?: string;
    num_segments?: number;
    quality: "draft" | "hd" | "fhd" | "4k";
    duration_mode: "auto" | "short" | "medium" | "long" | "cinematic";
    target_duration?: number | null;
    use_director: boolean;
    engine?: "auto" | "wan" | "ltx" | "cloud" | "studio" | "cogvideox" | "animatediff";
    aspect_ratio: "16:9" | "9:16" | "1:1";
    eco_mode: boolean;
    camera_movement: "static" | "dolly_in" | "dolly_out" | "pan_left" | "pan_right" | "drone_soar" | "orbit_right" | "tilt_up" | "tilt_down" | "handheld";
}

export const DEFAULT_PARAMS: GenerationParams = {
    dream_steps: 30,
    num_inference_steps: 30,
    motion_bucket_id: 65,
    fps: 16,
    guidance_scale: 5.0,
    seed: null,
    interpolate: true,
    audio_mode: "narration",
    negative_prompt: "deformed, distorted, disfigured, poorly drawn, bad anatomy, bad proportions, extra limbs, ugly, blurry, low quality, watermark",
    num_segments: 1,
    quality: "fhd",
    duration_mode: "auto",
    target_duration: null,
    use_director: false,
    engine: "wan",
    aspect_ratio: "16:9",
    eco_mode: false,
    camera_movement: "static",
};

interface AdvancedControlsProps {
    params: GenerationParams;
    onChange: (params: GenerationParams) => void;
    disabled?: boolean;
}

function SliderRow({
    label,
    value,
    min,
    max,
    step = 1,
    format = (v: number) => String(v),
    onChange,
    disabled,
}: {
    label: string;
    value: number;
    min: number;
    max: number;
    step?: number;
    format?: (v: number) => string;
    onChange: (v: number) => void;
    disabled?: boolean;
}) {
    const pct = ((value - min) / (max - min)) * 100;
    return (
        <div className="space-y-1.5">
            <div className="flex justify-between text-xs">
                <span className="text-white/50">{label}</span>
                <span className="font-mono text-purple-300">{format(value)}</span>
            </div>
            <input
                type="range"
                min={min}
                max={max}
                step={step}
                value={value}
                disabled={disabled}
                onChange={(e) => onChange(Number(e.target.value))}
                className={cn(
                    "w-full h-1.5 rounded-full appearance-none cursor-pointer",
                    "bg-gradient-to-r from-purple-500 to-blue-500/40",
                    "[&::-webkit-slider-thumb]:appearance-none",
                    "[&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:h-3.5",
                    "[&::-webkit-slider-thumb]:rounded-full",
                    "[&::-webkit-slider-thumb]:bg-white",
                    "[&::-webkit-slider-thumb]:shadow-[0_0_8px_rgba(168,85,247,0.8)]",
                    "[&::-webkit-slider-thumb]:cursor-pointer",
                    disabled && "opacity-50 cursor-not-allowed"
                )}
                style={{
                    background: `linear-gradient(to right, #a855f7 ${pct}%, rgba(59,130,246,0.2) ${pct}%)`,
                }}
            />
        </div>
    );
}

export function AdvancedControls({ params, onChange, disabled }: AdvancedControlsProps) {
    const [isOpen, setIsOpen] = useState(false);
    const [seedLocked, setSeedLocked] = useState(params.seed !== null);

    const update = (patch: Partial<GenerationParams>) => onChange({ ...params, ...patch });

    // ── Client-side estimated time calculation (matches director.py formula)
    const estimatedSeconds = (() => {
        const numScenes = params.use_director
            ? Math.max(1, Math.ceil((params.target_duration ?? ({
                auto: 7, short: 7, medium: 21, long: 60, cinematic: 90
              }[params.duration_mode] ?? 7)) / 7))
            : (params.num_segments ?? 1);
        const steps = params.num_inference_steps;
        const upscaleCost = { draft: 0.5, hd: 1.5, fhd: 3.0, "4k": 8.5 }[params.quality] ?? 3.0;
        let perScene = 35.0;
        if (params.engine === "cloud") {
            perScene = 20.0;
        } else if (params.engine === "wan") {
            perScene = 75.0 + upscaleCost;
        } else if (params.engine === "ltx") {
            perScene = 28.0 + upscaleCost;
        } else if (params.engine === "studio") {
            perScene = 720.0 + upscaleCost + (params.interpolate ? 5.0 : 0.5);
        } else {
            perScene = 3.0 + steps * 1.8 + (params.interpolate ? 5.0 : 0.5) + upscaleCost;
        }
        return Math.round(numScenes * perScene + (params.audio_mode !== "none" ? 4.0 : 0.5));
    })();

    const formatTime = (s: number) => {
        const m = Math.floor(s / 60);
        const sec = s % 60;
        return m > 0 ? `${m}dk ${sec.toString().padStart(2, "0")}sn` : `${sec}sn`;
    };

    const handleSeedToggle = () => {
        if (seedLocked) {
            setSeedLocked(false);
            update({ seed: null });
        } else {
            const randomSeed = Math.floor(Math.random() * 2 ** 32);
            setSeedLocked(true);
            update({ seed: randomSeed });
        }
    };

    const audioOptions: { value: GenerationParams["audio_mode"]; icon: React.ReactNode; label: string; desc: string }[] = [
        { value: "none", icon: <VolumeX className="w-3.5 h-3.5" />, label: "Silent", desc: "No audio" },
        { value: "foley", icon: <Sparkles className="w-3.5 h-3.5 text-amber-400" />, label: "Foley / SFX", desc: "Auto scene sound" },
        { value: "music", icon: <Music className="w-3.5 h-3.5" />, label: "Music", desc: "MusicGen ambient" },
        { value: "narration", icon: <Mic className="w-3.5 h-3.5" />, label: "Narration", desc: "Kokoro TTS" },
    ];

    return (
        <div className="w-full max-w-2xl">
            {/* Toggle Button */}
            <button
                onClick={() => setIsOpen((v) => !v)}
                disabled={disabled}
                className={cn(
                    "w-full flex items-center justify-between",
                    "px-4 py-2.5 rounded-xl",
                    "bg-white/5 hover:bg-white/8 border border-white/10",
                    "text-sm text-white/70 hover:text-white/90",
                    "transition-all duration-200",
                    disabled && "opacity-40 cursor-not-allowed"
                )}
                id="advanced-controls-toggle"
            >
                <span className="flex items-center gap-2">
                        <SlidersHorizontal className="w-4 h-4 text-purple-400" />
                        Advanced Controls
                        {/* Quick summary badges */}
                        <span className="hidden sm:flex items-center gap-1.5 ml-2">
                            <Badge>{params.dream_steps}d/{params.num_inference_steps}a steps</Badge>
                            {params.interpolate && <Badge className="text-blue-300 border-blue-500/30">24fps</Badge>}
                            <Badge className="text-cyan-300 border-cyan-500/30">
                                {params.quality.toUpperCase()}
                            </Badge>
                            <Badge className="text-orange-300 border-orange-500/30">
                                <Clock className="w-2.5 h-2.5 inline mr-0.5" />
                                ~{formatTime(estimatedSeconds)}
                            </Badge>
                            {params.audio_mode !== "none" && (
                                <Badge className="text-green-300 border-green-500/30">
                                    {params.audio_mode}
                                </Badge>
                            )}
                            {params.seed !== null && <Badge className="text-yellow-300 border-yellow-500/30">🔒 seed</Badge>}
                        </span>
                    </span>
                <ChevronDown
                    className={cn(
                        "w-4 h-4 text-white/40 transition-transform duration-200",
                        isOpen && "rotate-180"
                    )}
                />
            </button>

            {/* Controls Panel */}
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        transition={{ duration: 0.22, ease: "easeInOut" }}
                        className="overflow-hidden"
                    >
                        <div className="mt-2 p-4 rounded-xl bg-white/[0.04] border border-white/10 backdrop-blur-sm space-y-5">

                            {/* ── Estimated Time Banner ── */}
                            <div className="flex items-center justify-between px-3 py-2 rounded-lg bg-gradient-to-r from-purple-500/10 to-blue-500/10 border border-purple-500/20">
                                <div className="flex items-center gap-2 text-xs text-white/60">
                                    <Clock className="w-3.5 h-3.5 text-purple-400" />
                                    <span>Tahmini İşlem Süresi</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className="text-sm font-mono font-semibold text-purple-300">
                                        ~{formatTime(estimatedSeconds)}
                                    </span>
                                    <span className="text-[10px] text-white/30">(RTX 4060)</span>
                                </div>
                            </div>

                            {/* ── AI Engine Selection Section ── */}
                            <div className="space-y-3">
                                <SectionLabel icon={<Sparkles className="w-3.5 h-3.5 text-amber-400" />} label="AI Generation Engine" />
                                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-1.5">
                                    {([
                                        { id: "wan" as const, label: "Wan2.1", badge: "SOTA DiT", desc: "Alibaba Sinema" },
                                        { id: "ltx" as const, label: "LTX-Video", badge: "24fps DiT", desc: "Ultra Hızlı" },
                                        { id: "cloud" as const, label: "Cloud Studio", badge: "Veo 2 / Kling", desc: "Hollywood API" },
                                        { id: "studio" as const, label: "Studio (Old)", badge: "SDXL+SVD", desc: "Eski Motor" },
                                        { id: "cogvideox" as const, label: "CogVideoX", badge: "DiT 2B", desc: "Fotorealistik" },
                                        { id: "animatediff" as const, label: "AnimateDiff", badge: "8-Step", desc: "Hızlı / Anime" },
                                    ] as const).map(({ id, label, badge, desc }) => (
                                        <button
                                            key={id}
                                            disabled={disabled}
                                            onClick={() => update({ engine: id })}
                                            className={cn(
                                                "py-2 px-1.5 rounded-lg border text-center transition-all",
                                                (params.engine || "wan") === id
                                                    ? "bg-gradient-to-b from-amber-500/25 to-purple-500/15 border-amber-500/60 text-amber-200 shadow-[0_0_12px_rgba(245,158,11,0.3)]"
                                                    : "bg-white/5 border-white/10 text-white/50 hover:border-white/25"
                                            )}
                                            id={`engine-${id}`}
                                        >
                                            <div className="text-[11px] font-bold">{label}</div>
                                            <div className="text-[9px] font-mono text-amber-400/90">{badge}</div>
                                            <div className="text-[9px] opacity-50 mt-0.5">{desc}</div>
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <Divider />

                            {/* ── Quality Preset Section ── */}
                            <div className="space-y-3">
                                <SectionLabel icon={<Maximize2 className="w-3.5 h-3.5" />} label="Output Quality & Resolution" />
                                <div className="grid grid-cols-4 gap-1.5">
                                    {([
                                        { q: "draft" as const, label: "Draft", res: "576p", desc: "Hızlı" },
                                        { q: "hd" as const, label: "HD", res: "720p", desc: "Dengeli" },
                                        { q: "fhd" as const, label: "FHD", res: "1080p", desc: "Sinema" },
                                        { q: "4k" as const, label: "4K", res: "2160p", desc: "Ultra" },
                                    ] as const).map(({ q, label, res, desc }) => (
                                        <button
                                            key={q}
                                            disabled={disabled}
                                            onClick={() => update({ quality: q })}
                                            className={cn(
                                                "py-2 px-1 rounded-lg border text-center transition-all",
                                                params.quality === q
                                                    ? "bg-gradient-to-b from-purple-500/25 to-blue-500/15 border-purple-500/60 text-purple-200 shadow-[0_0_12px_rgba(168,85,247,0.3)]"
                                                    : "bg-white/5 border-white/10 text-white/50 hover:border-white/25"
                                            )}
                                            id={`quality-${q}`}
                                        >
                                            <div className="text-[11px] font-bold">{label}</div>
                                            <div className="text-[9px] font-mono opacity-70">{res}</div>
                                            <div className="text-[9px] opacity-50 mt-0.5">{desc}</div>
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <Divider />

                            {/* ── Camera Movement Section ── */}
                            <div className="space-y-3">
                                <SectionLabel icon={<Video className="w-3.5 h-3.5 text-violet-400" />} label="Kamera Hareketi / Camera Movement" />
                                <div className="grid grid-cols-5 gap-1">
                                    {([
                                        { id: "static"      as const, label: "Sabit",      emoji: "📷" },
                                        { id: "dolly_in"    as const, label: "Dolly İçi",  emoji: "🎯" },
                                        { id: "dolly_out"   as const, label: "Dolly Dışı", emoji: "↔️" },
                                        { id: "pan_left"    as const, label: "Sol Pan",    emoji: "⬅️" },
                                        { id: "pan_right"   as const, label: "Sağ Pan",    emoji: "➡️" },
                                        { id: "drone_soar"  as const, label: "Drone",      emoji: "🚁" },
                                        { id: "orbit_right" as const, label: "Orbit",      emoji: "🔄" },
                                        { id: "tilt_up"     as const, label: "Yukarı",     emoji: "⬆️" },
                                        { id: "tilt_down"   as const, label: "Aşağı",      emoji: "⬇️" },
                                        { id: "handheld"    as const, label: "El Kamerası", emoji: "🤳" },
                                    ] as const).map(({ id, label, emoji }) => (
                                        <button
                                            key={id}
                                            disabled={disabled}
                                            onClick={() => update({ camera_movement: id })}
                                            className={cn(
                                                "py-1.5 px-1 rounded-lg border text-center transition-all flex flex-col items-center justify-center gap-0.5",
                                                (params.camera_movement || "static") === id
                                                    ? "bg-gradient-to-b from-violet-500/25 to-purple-500/15 border-violet-500/60 text-violet-200 shadow-[0_0_12px_rgba(139,92,246,0.3)]"
                                                    : "bg-white/5 border-white/10 text-white/50 hover:border-white/25"
                                            )}
                                            id={`camera-${id}`}
                                        >
                                            <span className="text-base leading-none">{emoji}</span>
                                            <span className="text-[9px] font-medium leading-tight">{label}</span>
                                        </button>
                                    ))}
                                </div>
                                {(params.camera_movement && params.camera_movement !== "static") && (
                                    <p className="text-[10px] text-violet-400/70 px-1">
                                        ✨ Kamera yönü prompt&apos;a otomatik eklenir — Wan2.1 bunu sinematik hareket olarak yorumlar
                                    </p>
                                )}
                            </div>

                            <Divider />

                            {/* ── Aspect Ratio Section ── */}
                            <div className="space-y-3">
                                <SectionLabel icon={<Film className="w-3.5 h-3.5 text-cyan-400" />} label="Aspect Ratio / Ekran Oranı" />
                                <div className="grid grid-cols-3 gap-1.5">
                                    {([
                                        { id: "16:9" as const, label: "16:9 Sinema", res: "1920x1080", icon: <Monitor className="w-3 h-3 text-cyan-400" /> },
                                        { id: "9:16" as const, label: "9:16 Reels", res: "1080x1920", icon: <Smartphone className="w-3 h-3 text-pink-400" /> },
                                        { id: "1:1" as const, label: "1:1 Kare", res: "1080x1080", icon: <Square className="w-3 h-3 text-amber-400" /> },
                                    ] as const).map(({ id, label, res, icon }) => (
                                        <button
                                            key={id}
                                            disabled={disabled}
                                            onClick={() => update({ aspect_ratio: id })}
                                            className={cn(
                                                "py-2 px-1.5 rounded-lg border text-center transition-all flex flex-col items-center justify-center gap-0.5",
                                                (params.aspect_ratio || "16:9") === id
                                                    ? "bg-gradient-to-b from-cyan-500/25 to-blue-500/15 border-cyan-500/60 text-cyan-200 shadow-[0_0_12px_rgba(6,182,212,0.3)]"
                                                    : "bg-white/5 border-white/10 text-white/50 hover:border-white/25"
                                            )}
                                            id={`aspect-${id}`}
                                        >
                                            <div className="flex items-center gap-1 text-[11px] font-bold">
                                                {icon}
                                                <span>{label}</span>
                                            </div>
                                            <div className="text-[9px] font-mono opacity-50">{res}</div>
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <Divider />

                            {/* ── Eco Mode (Silent Background Hardware Protection) ── */}
                            <div className="flex items-center justify-between py-1 px-3 rounded-lg bg-emerald-500/5 border border-emerald-500/20">
                                <div className="flex items-center gap-2.5">
                                    <div className="p-1.5 rounded-md bg-emerald-500/15 text-emerald-400">
                                        <Leaf className="w-4 h-4" />
                                    </div>
                                    <div className="text-xs">
                                        <p className="text-white/80 font-medium">Sessiz Mod (Eco Mode)</p>
                                        <p className="text-emerald-400/70 text-[10px]">Fan sesini ve ısınmayı önler, bilgisayarı kastırmadan üretir</p>
                                    </div>
                                </div>
                                <Toggle
                                    checked={params.eco_mode}
                                    onChange={(v) => update({ eco_mode: v })}
                                    disabled={disabled}
                                    id="eco-mode-toggle"
                                />
                            </div>

                            <Divider />

                            {/* ── AI Director Section ── */}
                            <div className="space-y-3">
                                <SectionLabel icon={<Cpu className="w-3.5 h-3.5 text-cyan-400" />} label="AI Director — Uzun Video Planlayıcı" />

                                {/* Director toggle */}
                                <div className="flex items-center justify-between py-1">
                                    <div className="text-xs">
                                        <p className="text-white/70">AI Yönetmen Modu</p>
                                        <p className="text-white/30 text-[10px] mt-0.5">Promptu analiz ederek çok sahneli storyboard planlar</p>
                                    </div>
                                    <Toggle
                                        checked={params.use_director}
                                        onChange={(v) => update({ use_director: v })}
                                        disabled={disabled}
                                        id="director-toggle"
                                    />
                                </div>

                                {/* Duration Mode (visible when director is on OR always useful) */}
                                <div className="space-y-1.5">
                                    <span className="text-xs text-white/50">Süre Modu</span>
                                    <div className="grid grid-cols-5 gap-1">
                                        {([
                                            { mode: "auto" as const, label: "Auto", dur: "≈prompt" },
                                            { mode: "short" as const, label: "Kısa", dur: "~7sn" },
                                            { mode: "medium" as const, label: "Orta", dur: "~21sn" },
                                            { mode: "long" as const, label: "Uzun", dur: "~60sn" },
                                            { mode: "cinematic" as const, label: "Sinema", dur: "~90sn" },
                                        ] as const).map(({ mode, label, dur }) => (
                                            <button
                                                key={mode}
                                                disabled={disabled}
                                                onClick={() => update({ duration_mode: mode })}
                                                className={cn(
                                                    "py-1.5 px-1 rounded-lg border text-center transition-all",
                                                    params.duration_mode === mode
                                                        ? "bg-cyan-500/20 border-cyan-500/50 text-cyan-300 shadow-[0_0_8px_rgba(6,182,212,0.3)]"
                                                        : "bg-white/5 border-white/10 text-white/50 hover:border-white/20"
                                                )}
                                                id={`duration-mode-${mode}`}
                                            >
                                                <div className="text-[10px] font-medium">{label}</div>
                                                <div className="text-[8px] opacity-60">{dur}</div>
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                {/* Manual target duration */}
                                <div className="space-y-1.5">
                                    <div className="flex justify-between text-xs">
                                        <span className="text-white/50">Hedef Süre (sn) — isteğe bağlı</span>
                                        <span className="text-white/30 text-[10px]">4–180 saniye</span>
                                    </div>
                                    <input
                                        type="number"
                                        min={4}
                                        max={180}
                                        placeholder="Boş = Süre moduna göre"
                                        value={params.target_duration ?? ""}
                                        disabled={disabled}
                                        onChange={(e) => {
                                            const v = parseInt(e.target.value, 10);
                                            update({ target_duration: isNaN(v) ? null : Math.min(Math.max(v, 4), 180) });
                                        }}
                                        className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-white/80 placeholder:text-white/20 focus:outline-none focus:border-cyan-500/50"
                                        id="target-duration-input"
                                    />
                                </div>
                            </div>

                            <Divider />

                            {/* ── Dreamer Section ── */}
                            <div className="space-y-3">
                                <SectionLabel icon={<Zap className="w-3.5 h-3.5" />} label="Dreamer (SD 1.5 — Base Frame)" />
                                <SliderRow
                                    label="Dream Steps"
                                    value={params.dream_steps}
                                    min={10}
                                    max={50}
                                    format={(v) => `${v} steps`}
                                    onChange={(v) => update({ dream_steps: v })}
                                    disabled={disabled}
                                />
                                <SliderRow
                                    label="Guidance Scale (CFG)"
                                    value={params.guidance_scale}
                                    min={1}
                                    max={20}
                                    step={0.5}
                                    format={(v) => v.toFixed(1)}
                                    onChange={(v) => update({ guidance_scale: v })}
                                    disabled={disabled}
                                />
                                {/* Negative Prompt Input */}
                                <div className="space-y-1.5">
                                    <div className="flex justify-between text-xs">
                                        <span className="text-white/50">Negative Prompt</span>
                                        <span className="text-white/30 text-[10px]">Prevents deformities & blur</span>
                                    </div>
                                    <input
                                        type="text"
                                        value={params.negative_prompt ?? ""}
                                        placeholder="deformed, distorted, blurry, bad anatomy, low quality..."
                                        disabled={disabled}
                                        onChange={(e) => update({ negative_prompt: e.target.value })}
                                        className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-white/80 placeholder:text-white/20 focus:outline-none focus:border-purple-500/50"
                                    />
                                </div>
                            </div>

                            <Divider />

                            {/* ── Animator Section ── */}
                            <div className="space-y-3">
                                <SectionLabel icon={<Footprints className="w-3.5 h-3.5" />} label="Animator (SVD-XT — Motion)" />
                                
                                {/* Video Duration / Chaining */}
                                <div className="space-y-1.5">
                                    <div className="flex justify-between text-xs">
                                        <span className="text-white/50 flex items-center gap-1">
                                            <Film className="w-3 h-3 text-purple-400" /> Video Length & Segments
                                        </span>
                                        <span className="text-purple-300 font-mono text-[10px]">
                                            {(params.num_segments ?? 1) === 1 && "~3.5s (25 frames)"}
                                            {(params.num_segments ?? 1) === 2 && "~7.0s (49 frames - Extended)"}
                                            {(params.num_segments ?? 1) === 3 && "~10.5s (73 frames - Long)"}
                                            {(params.num_segments ?? 1) === 4 && "~14.0s (97 frames - Cinema)"}
                                        </span>
                                    </div>
                                    <div className="grid grid-cols-4 gap-1.5">
                                        {[
                                            { seg: 1, label: "Short", dur: "~3.5s" },
                                            { seg: 2, label: "Medium", dur: "~7s" },
                                            { seg: 3, label: "Long", dur: "~11s" },
                                            { seg: 4, label: "Cinema", dur: "~14s" },
                                        ].map(({ seg, label, dur }) => (
                                            <button
                                                key={seg}
                                                disabled={disabled}
                                                onClick={() => update({ num_segments: seg })}
                                                className={cn(
                                                    "py-1.5 px-1 rounded-lg border text-center transition-all",
                                                    (params.num_segments ?? 1) === seg
                                                        ? "bg-purple-500/20 border-purple-500/50 text-purple-300 shadow-[0_0_8px_rgba(168,85,247,0.3)]"
                                                        : "bg-white/5 border-white/10 text-white/50 hover:border-white/20"
                                                )}
                                            >
                                                <div className="text-[11px] font-medium">{label}</div>
                                                <div className="text-[9px] opacity-60">{dur}</div>
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                <SliderRow
                                    label="Animator Steps"
                                    value={params.num_inference_steps}
                                    min={5}
                                    max={50}
                                    format={(v) => `${v} steps`}
                                    onChange={(v) => update({ num_inference_steps: v })}
                                    disabled={disabled}
                                />
                                <SliderRow
                                    label="Motion Intensity"
                                    value={params.motion_bucket_id}
                                    min={1}
                                    max={255}
                                    format={(v) => `${v}/255 (sweet spot: 50-80)`}
                                    onChange={(v) => update({ motion_bucket_id: v })}
                                    disabled={disabled}
                                />
                                {/* FPS selector */}
                                <div className="space-y-1.5">
                                    <div className="flex justify-between text-xs">
                                        <span className="text-white/50 flex items-center gap-1">
                                            <Gauge className="w-3 h-3" /> Native FPS
                                        </span>
                                        <span className="text-white/40 text-[10px]">
                                            {params.interpolate ? `→ ${params.fps}fps native, interpolated to 24fps` : `${params.fps}fps output`}
                                        </span>
                                    </div>
                                    <div className="flex gap-2">
                                        {[7, 14].map((f) => (
                                            <button
                                                key={f}
                                                disabled={disabled}
                                                onClick={() => update({ fps: f })}
                                                className={cn(
                                                    "flex-1 py-1.5 text-xs rounded-lg border transition-all",
                                                    params.fps === f
                                                        ? "bg-purple-500/20 border-purple-500/40 text-purple-300"
                                                        : "bg-white/5 border-white/10 text-white/50 hover:border-white/20"
                                                )}
                                            >
                                                {f} fps
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            </div>

                            <Divider />

                            {/* ── Output Enhancement Section ── */}
                            <div className="space-y-3">
                                <SectionLabel icon={<Zap className="w-3.5 h-3.5 text-blue-400" />} label="Output Enhancement" />

                                {/* Interpolation toggle */}
                                <div className="flex items-center justify-between py-1">
                                    <div className="text-xs">
                                        <p className="text-white/70">Frame Interpolation (24fps)</p>
                                        <p className="text-white/30 text-[10px] mt-0.5">FFmpeg minterpolate: 14→56 frames</p>
                                    </div>
                                    <Toggle
                                        checked={params.interpolate}
                                        onChange={(v) => update({ interpolate: v })}
                                        disabled={disabled}
                                        id="interpolate-toggle"
                                    />
                                </div>

                                {/* Audio Mode */}
                                <div className="space-y-1.5">
                                    <span className="text-xs text-white/50">Audio Synthesis</span>
                                    <div className="grid grid-cols-3 gap-1.5">
                                        {audioOptions.map(({ value, icon, label, desc }) => (
                                            <button
                                                key={value}
                                                disabled={disabled}
                                                onClick={() => update({ audio_mode: value })}
                                                className={cn(
                                                    "py-2 px-2 rounded-lg border text-center transition-all",
                                                    params.audio_mode === value
                                                        ? "bg-purple-500/15 border-purple-500/40 text-purple-300"
                                                        : "bg-white/5 border-white/10 text-white/40 hover:border-white/20 hover:text-white/60"
                                                )}
                                                id={`audio-mode-${value}`}
                                            >
                                                <div className="flex justify-center mb-1">{icon}</div>
                                                <div className="text-[11px] font-medium">{label}</div>
                                                <div className="text-[9px] opacity-60 mt-0.5">{desc}</div>
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            </div>

                            <Divider />

                            {/* ── Seed Control ── */}
                            <div className="space-y-2">
                                <SectionLabel icon={<Dices className="w-3.5 h-3.5" />} label="Reproducibility" />
                                <div className="flex items-center gap-2">
                                    <input
                                        type="number"
                                        value={params.seed ?? ""}
                                        placeholder="Random seed"
                                        disabled={!seedLocked || disabled}
                                        onChange={(e) => {
                                            const v = parseInt(e.target.value, 10);
                                            if (!isNaN(v) && v >= 0 && v <= 4294967295) {
                                                update({ seed: v });
                                            }
                                        }}
                                        className={cn(
                                            "flex-1 bg-white/5 border border-white/10 rounded-lg px-3 py-1.5",
                                            "text-xs font-mono text-white/70 focus:outline-none focus:border-purple-500/50",
                                            "placeholder:text-white/20",
                                            (!seedLocked || disabled) && "opacity-50 cursor-not-allowed"
                                        )}
                                        id="seed-input"
                                    />
                                    <button
                                        onClick={handleSeedToggle}
                                        disabled={disabled}
                                        title={seedLocked ? "Unlock seed (random)" : "Lock seed for reproducibility"}
                                        className={cn(
                                            "p-2 rounded-lg border transition-all",
                                            seedLocked
                                                ? "bg-yellow-500/15 border-yellow-500/30 text-yellow-400"
                                                : "bg-white/5 border-white/10 text-white/40 hover:text-white/60"
                                        )}
                                        id="seed-lock-btn"
                                    >
                                        {seedLocked ? <Lock className="w-3.5 h-3.5" /> : <Unlock className="w-3.5 h-3.5" />}
                                    </button>
                                </div>
                                {!seedLocked && (
                                    <p className="text-[10px] text-white/30">
                                        A random seed is assigned per generation. Lock to reproduce exact results.
                                    </p>
                                )}
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

// ── Small helpers ────────────────────────────────────────────────────────────

function Badge({ children, className }: { children: React.ReactNode; className?: string }) {
    return (
        <span
            className={cn(
                "px-1.5 py-0.5 rounded border text-[9px] font-mono",
                "text-white/40 border-white/10 bg-white/5",
                className
            )}
        >
            {children}
        </span>
    );
}

function SectionLabel({ icon, label }: { icon: React.ReactNode; label: string }) {
    return (
        <div className="flex items-center gap-1.5 text-xs text-white/40 font-medium pb-1 border-b border-white/5">
            <span className="text-purple-400/70">{icon}</span>
            {label}
        </div>
    );
}

function Divider() {
    return <div className="border-t border-white/5" />;
}

function Toggle({
    checked,
    onChange,
    disabled,
    id,
}: {
    checked: boolean;
    onChange: (v: boolean) => void;
    disabled?: boolean;
    id?: string;
}) {
    return (
        <button
            id={id}
            role="switch"
            aria-checked={checked}
            disabled={disabled}
            onClick={() => onChange(!checked)}
            className={cn(
                "relative inline-flex h-5 w-9 items-center rounded-full transition-colors duration-200",
                "focus:outline-none focus-visible:ring-2 focus-visible:ring-purple-500",
                checked ? "bg-purple-500" : "bg-white/15",
                disabled && "opacity-50 cursor-not-allowed"
            )}
        >
            <span
                className={cn(
                    "inline-block h-3.5 w-3.5 rounded-full bg-white shadow-sm",
                    "transform transition-transform duration-200",
                    checked ? "translate-x-4.5" : "translate-x-0.5"
                )}
            />
        </button>
    );
}
