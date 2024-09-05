from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTasks
import shutil
from celery_tasks import detect_object
from celery.result import AsyncResult
import os
from fastapi.staticfiles import StaticFiles
import uuid

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/temp", StaticFiles(directory="temp"), name="temp")

temp_dir = "temp"

@app.get("/", response_class=HTMLResponse)
async def get_home():
    return templates.TemplateResponse("index.html", {"request": {}})

@app.post("/detect")
async def detect(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    # Đảm bảo thư mục 'temp/' tồn tại
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Tạo tên file duy nhất
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_location = os.path.join(temp_dir, unique_filename)
    
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)
        
    # Chạy Celery để phát hiện object
    task = detect_object.delay(file_location)

    # Trả về ID của tác vụ và đường dẫn tới ảnh gốc
    return {
        "task_id": task.id,
        "original_image_url": f"/temp/{unique_filename}"  # Trả về URL ảnh gốc
    }

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    task_result = AsyncResult(task_id)
    
    if task_result.state == 'PENDING':
        return {"status": "Processing..."}
    elif task_result.state != 'FAILURE':
        if 'output_image' in task_result.result:
            output_image = task_result.result['output_image']
            return {
                "status": "Completed",
                "output_image_url": f"/{temp_dir}/{os.path.basename(output_image)}"  # URL ảnh kết quả
            }
        else:
            return {"status": task_result.state, "result": task_result.result}
    else:
        return {"status": "Failed", "result": str(task_result.info)}
