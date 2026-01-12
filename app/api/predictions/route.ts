import { NextResponse } from "next/server";
import Replicate from "replicate";

const replicate = new Replicate({
    auth: process.env.REPLICATE_API_TOKEN,
});

export async function POST(request: Request) {
    if (!process.env.REPLICATE_API_TOKEN) {
        return NextResponse.json(
            { error: "REPLICATE_API_TOKEN is not configured" },
            { status: 500 }
        );
    }

    try {
        const { prompt } = await request.json();

        const response = await replicate.predictions.create({
            // Stable Video Diffusion XT
            version: "3f0457e4619daac51203dedb472816f3af3126d47b649d2906e788c1cc9065a3",
            input: {
                prompt,
                video_length: "14 frames_with_svd_xt",
                sizing_strategy: "maintain_aspect_ratio",
                frames_per_second: 6,
                motion_bucket_id: 127,
            },
        });

        return NextResponse.json(response, { status: 201 });
    } catch (error) {
        console.error("Prediction error:", error);
        return NextResponse.json({ error: "Failed to generate video" }, { status: 500 });
    }
}
