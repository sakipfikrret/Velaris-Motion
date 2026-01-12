import { SiteHeader } from "@/components/site-header";
import { HeroSection } from "@/components/hero-section";

export default function Home() {
  return (
    <main className="min-h-screen bg-black overflow-x-hidden selection:bg-white/20">
      <SiteHeader />
      <HeroSection />
    </main>
  );
}
