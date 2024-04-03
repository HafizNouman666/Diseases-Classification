from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize TFSMLayer
tfsmlayer = tf.keras.layers.TFSMLayer("../saved_models/1", call_endpoint='serving_default')

# Create a Keras model with TFSMLayer as its only layer
MODEL = tf.keras.Sequential([
    tfsmlayer
])

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


def read_file_as_image(data) -> np.ndarray:
    try:
        image = np.array(Image.open(BytesIO(data)))
        return image
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file as image: {e}")


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    try:
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)

        predictions_dict = MODEL.predict(img_batch)

        # Log predictions for debugging
        print("Predictions:", predictions_dict)

        # Ensure predictions_dict is not empty
        if not predictions_dict:
            raise HTTPException(status_code=500, detail="No predictions returned")

        # Extract predictions for the output layer
        predictions = predictions_dict['dense_1']

        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = float(np.max(predictions))

        return {
            "class": predicted_class,
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
