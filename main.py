from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoImageProcessor, ViTForImageClassification
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from collections import deque
from contextlib import contextmanager
import torch
import os, json, io

app = FastAPI()
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
from datetime import datetime, timedelta
import time

allowed_origins = [
    "*",
    "http://localhost:8000",
    "https://isitbanana.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins, 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

METRICS_FILE = "metrics.json"
MAX_REQUESTS = 1000000  # Store last million requests


if os.path.exists(METRICS_FILE):
    with open(METRICS_FILE, "r") as f:
        loaded_metrics = json.load(f)
        metrics = {
            "total_requests": loaded_metrics["total_requests"],
            "requests_log": deque(loaded_metrics["requests_log"], maxlen=MAX_REQUESTS),
            "path_counts": loaded_metrics["path_counts"]
        }
else:
    metrics = {
        "total_requests": 0,
        "requests_log": deque(maxlen=MAX_REQUESTS),
        "path_counts": {}
    }



def save_metrics():
    with open(METRICS_FILE, "w") as f:
        json.dump({
            "total_requests": metrics["total_requests"],
            "requests_log": list(metrics["requests_log"]),
            "path_counts": metrics["path_counts"]
        }, f)

@app.middleware("http")
async def log_request(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    metrics["total_requests"] += 1
    metrics["requests_log"].append({
        "timestamp": datetime.now().isoformat(),
        "method": request.method,
        "path": request.url.path,
        "process_time": process_time
    })
    
    path_key = f"{request.method} {request.url.path}"
    metrics["path_counts"][path_key] = metrics["path_counts"].get(path_key, 0) + 1
    
    # Save metrics every 1000 requests to reduce I/O operations
    if metrics["total_requests"] % 1000 == 0:
        save_metrics()
    
    return response


@app.get("/")
async def root():
    return True

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:   
        # Read the contents of the file
        contents = await file.read()
            
        # Open the image using PIL
        image = Image.open(io.BytesIO(contents))
            
        width, height = image.size
            
        inputs = image_processor(image, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits


        predicted_label = logits.argmax(-1).item()
        image_type = model.config.id2label[predicted_label]
        return JSONResponse(content={"type":image_type}
                            )
    except Exception as e:
        # Return an error response
        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "message": "An error occurred while processing the image",
                "detail": str(e)
            }
        )

@app.get("/health")
async def health_check():
    try:
        # Check if the model and processor are loaded
        if image_processor and model:
            # Perform a simple inference to ensure the model is working
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                _ = model(dummy_input)
            return {"status": "healthy", "message": "Server is running and model is loaded"}
        else:
            return {"status": "unhealthy", "message": "Model or processor not loaded"}
    except Exception as e:
        return {"status": "unhealthy", "message": f"Error occurred: {str(e)}"}


@app.get("/metrics", response_class=HTMLResponse)
async def get_metrics():
    now = datetime.now()
    
    # Calculate metrics for different time periods
    time_periods = {
        "Last Hour": timedelta(hours=1),
        "Last Day": timedelta(days=1),
        "Last Week": timedelta(weeks=1),
        "Last Month": timedelta(days=30)
    }
    
    period_metrics = {}
    for period_name, period_delta in time_periods.items():
        period_start = now - period_delta
        period_requests = [req for req in metrics["requests_log"] if datetime.fromisoformat(req["timestamp"]) > period_start]
        
        period_metrics[period_name] = {
            "requests": len(period_requests),
            "rpm": len(period_requests) / period_delta.total_seconds() * 60,
            "avg_response_time": sum(req["process_time"] for req in period_requests) / len(period_requests) if period_requests else 0
        }

    # Generate table rows
    table_rows = ""
    for period, data in period_metrics.items():
        table_rows += f"""
        <tr>
            <td data-label="Period">{period}</td>
            <td data-label="Requests">{data['requests']}</td>
            <td data-label="RPM">{data['rpm']:.2f}</td>
            <td data-label="Avg Response Time">{data['avg_response_time']:.3f}</td>
        </tr>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Metrics</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
         <style>
             body {{
                 font-family: 'Arial', sans-serif;
                 line-height: 1.6;
                 color: #333;
                 max-width: 1200px;
                 margin: 0 auto;
                 padding: 20px;
                 background-color: #f0f8ff;
             }}
             h1, h2 {{
                 color: #4a4a4a;
                 text-align: center;
             }}
             .metrics-container {{
                 background-color: white;
                 border-radius: 10px;
                 padding: 20px;
                 box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                 margin-bottom: 20px;
             }}
             table {{
                 width: 100%;
                 border-collapse: collapse;
                 margin-top: 20px;
             }}
             th, td {{
                 padding: 12px;
                 text-align: left;
                 border-bottom: 1px solid #ddd;
             }}
             th {{
                 background-color: #f2f2f2;
             }}
             tr:hover {{
                 background-color: #f5f5f5;
             }}
             .chart-container {{
                 max-width: 600px;
                 margin: 0 auto;
             }}
             @media (max-width: 768px) {{
                 table, tr, td {{
                     display: block;
                 }}
                 tr {{
                     margin-bottom: 10px;
                 }}
                 td {{
                     border: none;
                     position: relative;
                     padding-left: 50%;
                 }}
                 td:before {{
                     content: attr(data-label);
                     position: absolute;
                     left: 6px;
                     width: 45%;
                     padding-right: 10px;
                     white-space: nowrap;
                     font-weight: bold;
                 }}
             }}
         </style>
     </head>
     <body>

    </head>
    <body>
        <h1>🚀 Banana Metrics 📊</h1>
        
        <div class="metrics-container">
            <p>Total Requests: {metrics["total_requests"]}</p>
            <p>Stored Requests: {len(metrics["requests_log"])}</p>
        </div>

        <div class="metrics-container">
            <h2>Metrics by Time Period</h2>
            <table>
                <tr>
                    <th>Period</th>
                    <th>Requests</th>
                    <th>RPM</th>
                    <th>Avg Response Time (s)</th>
                </tr>
                {table_rows}
            </table>
        </div>

        <div class="metrics-container">
            <h2>Top 10 Requested Paths</h2>
            <div class="chart-container">
                <canvas id="pathChart"></canvas>
            </div>
        </div>

        <script>
            var ctx = document.getElementById('pathChart').getContext('2d');
            var sortedPaths = Object.entries({json.dumps(metrics["path_counts"])}).sort((a, b) => b[1] - a[1]).slice(0, 10);
            var chart = new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: sortedPaths.map(item => item[0]),
                    datasets: [{{
                        label: 'Requests per Path',
                        data: sortedPaths.map(item => item[1]),
                        backgroundColor: 'rgba(75, 192, 192, 0.6)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        y: {{
                            beginAtZero: true
                        }}
                    }},
                    plugins: {{
                        legend: {{
                            display: false
                        }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
