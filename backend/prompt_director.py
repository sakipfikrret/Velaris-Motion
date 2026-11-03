"""
Velaris Motion - Prompt Director & Smart Ideation Engine
Transforms simple user ideas (Turkish/English) into coherent, physically realistic,
award-winning cinematic scene descriptions across 3 directorial concepts.
Zero-strain CPU operation (<1ms execution time).
"""

import re
from typing import List, Dict, Any, Literal
from pydantic import BaseModel


class SceneConcept(BaseModel):
    id: str
    title: str
    category: str
    badge: str
    enhanced_prompt: str
    negative_prompt: str
    camera_movement: str
    lighting: str


def _clean_input(text: str) -> str:
    # Remove excessive punctuation and whitespace
    return re.sub(r"\s+", " ", text).strip()


def ideate_cinematic_scenes(user_prompt: str) -> List[SceneConcept]:
    """
    Generates 3 distinct, logically coherent cinematic scene concepts for a user prompt.
    Concept 1: National Geographic / 8k Photorealistic Master Portrait (Extreme Detail)
    Concept 2: Epic Dynamic Action & Motion Dynamics (Cinematic Flight / Tracking)
    Concept 3: Atmospheric Golden Hour / Moody Cinema (Award-Winning Lighting)
    """
    raw = _clean_input(user_prompt)
    if not raw:
        raw = "a majestic golden eagle"

    # Strict universal negative prompt to eliminate multi-head distortions, mutations, noise, and grain
    universal_neg = (
        "multiple heads, two heads, twin, double, duplicate head, extra beaks, two beaks, extra eyes, mutated, "
        "split face, four eyes, three eyes, cloned face, extra faces, extra wings, extra limbs, extra legs, deformed beak, "
        "cartoon, anime, illustration, painting, drawing, 3d render, sketch, cgi, plastic, synthetic, "
        "deformed, bad anatomy, bad proportions, blurry, pixelated, grainy, noise, compression artifacts, "
        "watermark, signature, text, out of focus, low quality, oversaturated"
    )

    clean_raw = re.sub(r'^(a|an|the)\s+', '', raw, flags=re.IGNORECASE).strip()
    lower_raw = clean_raw.lower()

    # Domain classification
    is_landscape = any(w in lower_raw for w in [
        "mountain", "valley", "forest", "lake", "ocean", "sea", "river", "desert",
        "sky", "clouds", "sunset", "sunrise", "hills", "canyon", "waterfall", "nature", "landscape", "tree", "pine"
    ])
    is_portrait = any(w in lower_raw for w in [
        "woman", "man", "person", "girl", "boy", "face", "portrait", "warrior", "samurai", "model"
    ])
    is_wildlife = any(w in lower_raw for w in [
        "eagle", "bird", "hawk", "falcon", "lion", "tiger", "wolf", "bear", "fox", "deer", "animal", "creature", "cat", "dog", "horse"
    ])

    if is_landscape:
        p1 = f"Breathtaking 8k National Geographic documentary landscape photograph of {clean_raw}, vast scenic panorama, atmospheric morning mist, Kodak Portra 400 natural film colors"
        p2 = f"Cinematic IMAX wide aerial tracking shot gliding over {clean_raw}, dramatic sweeping camera glide, volumetric sunlight rays, crisp geological and foliage detail"
        p3 = f"Moody golden hour fine-art landscape photograph of {clean_raw}, warm sunset glow casting long cinematic shadows, soft atmospheric haze, Roger Deakins style cinematography"
    elif is_portrait:
        p1 = f"Award-winning 8k photographic master portrait of a single {clean_raw}, centered, authentic skin texture with visible fine pores, soft natural catchlights, 85mm f/1.4 lens"
        p2 = f"Dynamic cinematic tracking shot of {clean_raw} in motion, authentic cinematic lighting, atmospheric background depth, 35mm film aesthetic, award-winning cinematography"
        p3 = f"Intimate golden hour cinematic portrait of {clean_raw}, warm amber rim-lighting, soft bokeh background, emotional depth, natural filmic dynamic range"
    elif is_wildlife:
        p1 = f"A breathtaking National Geographic wildlife documentary photograph of a single {clean_raw}, centered in frame, majestic single profile, ultra-sharp plumage or fur, piercing eye with crisp sunlight reflection, 85mm f/1.4 lens"
        p2 = f"Cinematic wide-angle tracking shot of a single majestic {clean_raw} moving through its natural habitat, fluid natural motion, volumetric atmospheric depth, filmed on 35mm IMAX camera"
        p3 = f"Atmospheric cinematic film still of a single majestic {clean_raw} at golden hour, centered in frame, warm sunset rim-lighting illuminating natural contours, soft environmental bokeh"
    else:
        p1 = f"Masterpiece 8k documentary photograph of a single {clean_raw}, centered composition, natural authentic textures, sharp optical focus, shot on 35mm lens, award-winning photography"
        p2 = f"Dynamic cinematic film tracking shot of {clean_raw}, fluid camera motion, dramatic atmospheric depth, professional cinematography, 8k resolution"
        p3 = f"Moody atmospheric cinematic scene featuring {clean_raw} at golden hour, warm amber illumination, soft background falloff, Roger Deakins style cinematography"

    # 1. National Geographic 8K Master Shot
    concept_1 = SceneConcept(
        id="natgeo_master",
        title="National Geographic Doğa Belgeseli (8K Portre)" if is_wildlife or is_landscape else "8K Sinematik Master Çekim",
        category="Doğa & Biyoloji" if is_wildlife or is_landscape else "Sinema & Belgesel",
        badge="NatGeo 8K",
        enhanced_prompt=p1,
        negative_prompt=universal_neg,
        camera_movement="Subtle slow push-in, shallow depth of field",
        lighting="Direct crisp sunlight with natural specular highlights",
    )

    # 2. Dynamic Cinematic Action
    concept_2 = SceneConcept(
        id="epic_action",
        title="Dinamik Sinematik Uçuş & Aksiyon" if is_wildlife else "Dinamik Sinematik Kamera Takibi",
        category="Sinematik Aksiyon",
        badge="Dinamik",
        enhanced_prompt=p2,
        negative_prompt=universal_neg,
        camera_movement="Dynamic fluid tracking shot following scene motion",
        lighting="Dramatic atmospheric volumetric light",
    )

    # 3. Golden Hour & Moody Cinematic Still
    concept_3 = SceneConcept(
        id="golden_hour",
        title="Altın Saat & Atmosferik Sinema (Golden Hour)",
        category="Sanatsal & Işık",
        badge="Altın Saat",
        enhanced_prompt=p3,
        negative_prompt=universal_neg,
        camera_movement="Slow cinematic pan left with gentle parallax",
        lighting="Warm golden hour rim light with soft twilight shadows",
    )

    return [concept_1, concept_2, concept_3]

