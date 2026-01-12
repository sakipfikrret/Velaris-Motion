from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from engine import HybridEngine

app = FastAPI(title="Velaris Motion AI Engine")

# Allow CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Engine globally (lazy loading inside)
engine = HybridEngine()

class GenerateRequest(BaseModel):
    prompt: str

@app.get("/")
def health_check():
    return {"status": "Velaris Engine is Ready", "device": "RTX 4060 (Optimized)"}

@app.post("/generate")
async def generate_video(req: GenerateRequest):
    try:
        # NOTE: In a real production app, this should be a background task (Celery/Redis).
        # For this local demo, we'll run it synchronously or use BackgroundTasks if we want to return an ID.
        # But to keep the UI simple for now, we wait. (Generation takes ~60s)
        
        video_path = engine.generate(req.prompt)
        
        # Return path relative to helper or serve directly
        filename = os.path.basename(video_path)
        return {"status": "succeeded", "video_url": f"http://localhost:8000/videos/{filename}"}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/videos/{filename}")
async def get_video(filename: str):
    video_path = os.path.join("outputs", filename)
    if os.path.exists(video_path):
        return FileResponse(video_path, media_type="video/mp4")
    return JSONResponse(status_code=404, content={"error": "Video not found"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
