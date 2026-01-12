import { NextResponse } from "next/server";
import Replicate from "replicate";

const replicate = new Replicate({
    auth: process.env.REPLICATE_API_TOKEN,
});

export async function GET(
    request: Request,
    { params }: { params: Promise<{ id: string }> }
) {
    const { id } = await params;
    if (!process.env.REPLICATE_API_TOKEN) {
        return NextResponse.json(
            { error: "REPLICATE_API_TOKEN is not configured" },
            { status: 500 }
        );
    }

    try {
        const prediction = await replicate.predictions.get(id);

        if (prediction?.error) {
            return NextResponse.json({ detail: prediction.error }, { status: 500 });
        }

        return NextResponse.json(prediction);
    } catch (error) {
        return NextResponse.json({ error: "Failed to fetch prediction" }, { status: 500 });
    }
}
