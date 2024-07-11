from transformers import AutoImageProcessor, ViTForImageClassification
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
app = FastAPI()


image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    # Read the contents of the file
    contents = await file.read()
    
    # Open the image using PIL
    image = Image.open(io.BytesIO(contents))
    
    # Process the image (example: get dimensions)
    width, height = image.size
    
    inputs = image_processor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits


    predicted_label = logits.argmax(-1).item()
    image_type = model.config.id2label[predicted_label]
    # You can perform more operations with PIL here
    return JSONResponse(content={
        "type":image_type
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

