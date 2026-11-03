import { NextResponse } from "next/server";

export async function GET(
    request: Request,
    { params }: { params: Promise<{ id: string }> }
) {
    const { id } = await params;

    try {
        const response = await fetch(`http://localhost:8000/jobs/${encodeURIComponent(id)}`);
        const data = await response.json();
        return NextResponse.json(data, { status: response.status });
    } catch (error: any) {
        return NextResponse.json(
            { error: "Failed to connect to local Velaris backend on port 8000." },
            { status: 502 }
        );
    }
}
