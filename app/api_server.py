from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import json

from app.main import run_analysis  # Make sure run_analysis is correctly defined

app = FastAPI()

# 🔧 Mount the /static path
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    # Serve the static index.html file
    file_path = os.path.join("static", "index.html")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)


@app.post("/analyze")
async def analyze_log_file(logfile: UploadFile = File(...)):
    temp_path = "temp_upload.log"
    temp_result_path = "temp_result.json"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(logfile.file, buffer)
    print(f"Saved to: {temp_path}")


    try:
        results = run_analysis(temp_path, temp_result_path)
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(temp_result_path):
            os.remove(temp_result_path)
