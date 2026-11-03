import Link from "next/link";
import { Sparkles } from "lucide-react";

export function SiteHeader() {
    return (
        <header className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 h-16 border-b border-white/5 bg-black/50 backdrop-blur-md">
            <Link href="/" className="flex items-center gap-2 group">
                <div className="p-1.5 rounded-lg bg-white/10 group-hover:bg-white/20 transition-colors">
                    <Sparkles className="w-5 h-5 text-white" />
                </div>
                <span className="font-semibold tracking-tight text-lg">Velaris Motion</span>
            </Link>

            <nav className="flex items-center gap-6 text-sm font-medium text-white/70">
                <Link href="/" className="hover:text-white transition-colors">Studio</Link>
                <Link href="/benchmark" className="hover:text-white transition-colors">Benchmark Lab</Link>
                <a href="http://localhost:8000/docs" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">API Docs</a>
            </nav>

            <div className="flex items-center gap-4">
                <button className="text-sm font-medium text-white/70 hover:text-white transition-colors">
                    Sign In
                </button>
                <button className="bg-white text-black px-4 py-2 rounded-full text-sm font-medium hover:bg-white/90 transition-colors">
                    Try Now
                </button>
            </div>
        </header>
    );
}
