"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    History,
    Play,
    Download,
    RotateCcw,
    Trash2,
    Music,
    Mic,
    VolumeX,
    Sparkles,
    Film,
    HardDrive,
    Clock,
} from "lucide-react";

export interface HistoryItem {
    id: string;
    prompt: string;
    videoUrl: string;
    createdAt: number;
    durationSeconds?: number;
    vramPeakMb?: number;
    fps?: number;
    seed?: number;
    audioMode?: "none" | "music" | "narration";
    isImg2Vid?: boolean;
    motionBucketId?: number;
    numInferenceSteps?: number;
}

interface GenerationHistoryProps {
    items: HistoryItem[];
    onReuse: (item: HistoryItem) => void;
    onDelete: (id: string) => void;
    onClearAll: () => void;
}

export function GenerationHistory({ items, onReuse, onDelete, onClearAll }: GenerationHistoryProps) {
    const [hoveredId, setHoveredId] = useState<string | null>(null);

    const formatTimestamp = (ts: number) => {
        const d = new Date(ts);
        return d.toLocaleDateString(undefined, {
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
        });
    };

    if (items.length === 0) {
        return null;
    }

    return (
        <section className="w-full max-w-5xl mx-auto px-4 mt-20 mb-12">
            <div className="flex items-center justify-between mb-6 pb-3 border-b border-white/10">
                <div className="flex items-center gap-3">
                    <div className="p-2 rounded-xl bg-purple-500/10 border border-purple-500/20 text-purple-400">
                        <History className="w-5 h-5" />
                    </div>
                    <div>
                        <h2 className="text-xl font-semibold tracking-tight text-white flex items-center gap-2">
                            Generation Vault
                            <span className="text-xs px-2 py-0.5 rounded-full bg-white/10 text-white/70 font-mono">
                                {items.length} {items.length === 1 ? "video" : "videos"}
                            </span>
                        </h2>
                        <p className="text-xs text-white/50">
                            Saved locally in your browser. Re-run or download anytime.
                        </p>
                    </div>
                </div>

                <button
                    onClick={onClearAll}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-white/40 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                >
                    <Trash2 className="w-3.5 h-3.5" />
                    Clear Vault
                </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
                <AnimatePresence>
                    {items.map((item) => {
                        const isHovered = hoveredId === item.id;
                        return (
                            <motion.div
                                key={item.id}
                                layout
                                initial={{ opacity: 0, y: 12, scale: 0.98 }}
                                animate={{ opacity: 1, y: 0, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.95 }}
                                transition={{ duration: 0.2 }}
                                onMouseEnter={() => setHoveredId(item.id)}
                                onMouseLeave={() => setHoveredId(null)}
                                className="group relative rounded-2xl border border-white/10 bg-white/[0.03] backdrop-blur-md overflow-hidden hover:border-purple-500/40 hover:shadow-xl hover:shadow-purple-500/5 transition-all flex flex-col justify-between"
                            >
                                {/* Video Preview */}
                                <div className="relative aspect-video bg-black/60 overflow-hidden">
                                    <video
                                        src={`http://localhost:8000${item.videoUrl}`}
                                        className="w-full h-full object-cover"
                                        loop
                                        muted
                                        playsInline
                                        autoPlay={isHovered}
                                    />

                                    {/* Overlay badge (Image-to-Video or Text-to-Video) */}
                                    <div className="absolute top-2.5 left-2.5 flex items-center gap-1.5">
                                        <span className="px-2 py-0.5 rounded-md text-[10px] font-mono tracking-wide bg-black/70 backdrop-blur-md text-white/80 border border-white/10 flex items-center gap-1">
                                            {item.isImg2Vid ? (
                                                <>
                                                    <Film className="w-3 h-3 text-cyan-400" />
                                                    IMG2VID
                                                </>
                                            ) : (
                                                <>
                                                    <Sparkles className="w-3 h-3 text-purple-400" />
                                                    TXT2VID
                                                </>
                                            )}
                                        </span>

                                        {item.audioMode && item.audioMode !== "none" && (
                                            <span className="px-2 py-0.5 rounded-md text-[10px] font-mono tracking-wide bg-black/70 backdrop-blur-md text-white/80 border border-white/10 flex items-center gap-1">
                                                {item.audioMode === "music" ? (
                                                    <>
                                                        <Music className="w-3 h-3 text-emerald-400" />
                                                        MUSIC
                                                    </>
                                                ) : (
                                                    <>
                                                        <Mic className="w-3 h-3 text-amber-400" />
                                                        TTS
                                                    </>
                                                )}
                                            </span>
                                        )}
                                    </div>

                                    {/* Seed Badge */}
                                    {item.seed !== undefined && (
                                        <div className="absolute top-2.5 right-2.5">
                                            <span className="px-2 py-0.5 rounded-md text-[10px] font-mono bg-black/70 backdrop-blur-md text-white/60 border border-white/10">
                                                #{item.seed}
                                            </span>
                                        </div>
                                    )}

                                    {/* Hover overlay play prompt */}
                                    {!isHovered && (
                                        <div className="absolute inset-0 flex items-center justify-center pointer-events-none bg-black/20">
                                            <div className="p-3 rounded-full bg-black/60 border border-white/20 text-white/80 backdrop-blur-sm group-hover:scale-110 transition-transform">
                                                <Play className="w-5 h-5 fill-white/80 translate-x-0.5" />
                                            </div>
                                        </div>
                                    )}
                                </div>

                                {/* Card Details */}
                                <div className="p-4 flex flex-col flex-1 justify-between gap-3">
                                    <div>
                                        <p className="text-xs text-white/90 line-clamp-2 font-medium leading-relaxed" title={item.prompt}>
                                            "{item.prompt}"
                                        </p>

                                        {/* Metadata row */}
                                        <div className="flex items-center gap-3 mt-2 text-[11px] text-white/40 font-mono">
                                            {item.durationSeconds !== undefined && (
                                                <span className="flex items-center gap-1">
                                                    <Clock className="w-3 h-3" />
                                                    {item.durationSeconds.toFixed(1)}s
                                                </span>
                                            )}
                                            {item.fps && (
                                                <span className="flex items-center gap-1">
                                                    <Film className="w-3 h-3" />
                                                    {item.fps} fps
                                                </span>
                                            )}
                                            {item.vramPeakMb !== undefined && item.vramPeakMb > 0 && (
                                                <span className="flex items-center gap-1">
                                                    <HardDrive className="w-3 h-3" />
                                                    {Math.round(item.vramPeakMb)} MB
                                                </span>
                                            )}
                                        </div>
                                    </div>

                                    {/* Action Bar */}
                                    <div className="flex items-center justify-between pt-2 border-t border-white/5 text-xs">
                                        <span className="text-[10px] text-white/30 font-mono">
                                            {formatTimestamp(item.createdAt)}
                                        </span>

                                        <div className="flex items-center gap-1">
                                            <button
                                                onClick={() => onReuse(item)}
                                                title="Reuse Prompt & Settings"
                                                className="p-1.5 rounded-lg text-white/60 hover:text-white hover:bg-white/10 transition-colors flex items-center gap-1 text-[11px]"
                                            >
                                                <RotateCcw className="w-3.5 h-3.5" />
                                                <span>Reuse</span>
                                            </button>

                                            <a
                                                href={`http://localhost:8000${item.videoUrl}`}
                                                download={`velaris_${item.id}.mp4`}
                                                title="Download MP4"
                                                className="p-1.5 rounded-lg text-white/60 hover:text-purple-400 hover:bg-purple-500/10 transition-colors"
                                            >
                                                <Download className="w-3.5 h-3.5" />
                                            </a>

                                            <button
                                                onClick={() => onDelete(item.id)}
                                                title="Delete from Vault"
                                                className="p-1.5 rounded-lg text-white/40 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                                            >
                                                <Trash2 className="w-3.5 h-3.5" />
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            </motion.div>
                        );
                    })}
                </AnimatePresence>
            </div>
        </section>
    );
}
