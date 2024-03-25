from fastapi import FastAPI, UploadFile, File, HTTPException
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

app = FastAPI()

# Load the TensorFlow models
plant_disease_model = tf.keras.models.load_model("models/plant_disease_model.h5")
soil_model = tf.keras.models.load_model("models/SoilNet_93_86.h5")
banana_model = tf.keras.models.load_model("models/banana2.h5")
sugarcane_model = tf.keras.models.load_model("models/sugarcane.h5")
corn_model = tf.keras.models.load_model("models/corn.hdf5")
coconut_model = tf.keras.models.load_model("models/coconut_model.h5")
paddy_model = tf.keras.models.load_model("models/ml_model.h5")
potato_model = tf.keras.models.load_model("models/potato.hdf5")
tomato_model = tf.keras.models.load_model("models/cnn_model.h5")
# Define class names for both models


plant_disease_class_names = ['early_rust', 'late_leaf_spot', 'nutrition_deficiency', 'healthy_leaf', 'early_leaf_spot', 'rust']
soil_class_names = ['Alluvial Soil', 'Black Soil', 'Clay Soil', 'Red Soil']
banana_class_names = ['cordana', 'healthy', 'pestalotiopsis', 'sigatoka']
coconut_class_names = ['Caterpillars' ,'Leaflets' ,'DryingofLeaflets' ,'Flaccidity' ,'Yellowing']
corn_class_names =  ['Blight','Common_Rust','Gray_Leaf_Spot','Healthy']
paddy_class_names = ['Leaf Blight', 'Brown Spot', 'Leaf Smut']
potato_class_names = ['Early_Blight','Healthy','Late_Blight']
sugarcane_class_names = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']
tomato_class_names = ['Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy'
]


@app.post("/predict/groundnut")
async def predict_groundnut(file: UploadFile = File(...)):
    try:
        img = Image.open(BytesIO(await file.read()))
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict plant disease
        plant_disease_predictions = plant_disease_model.predict(img_array)
        predicted_disease_class = plant_disease_class_names[np.argmax(plant_disease_predictions)]

        return {"class": predicted_disease_class, "predictions": plant_disease_predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during plant disease prediction: {e}")

@app.post("/predict/soil")
async def predict_soil(file: UploadFile = File(...)):
    try:
        img = Image.open(BytesIO(await file.read()))
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict soil type
        soil_predictions = soil_model.predict(img_array)
        predicted_soil_class = soil_class_names[np.argmax(soil_predictions)]

        return {"class": predicted_soil_class, "predictions": soil_predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during soil prediction: {e}")

@app.post("/predict/coconut")
async def predict_coconut(file: UploadFile = File(...)):
    try:
        img = Image.open(BytesIO(await file.read()))
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict soil type
        soil_predictions = coconut_model.predict(img_array)
        predicted_soil_class = coconut_class_names[np.argmax(soil_predictions)]

        return {"class": predicted_soil_class, "predictions": soil_predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during soil prediction: {e}")
@app.post("/predict/corn")
async def predict_corn(file: UploadFile = File(...)):
    try:
        img = Image.open(BytesIO(await file.read()))
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict soil type
        soil_predictions = corn_model.predict(img_array)
        predicted_soil_class = corn_class_names[np.argmax(soil_predictions)]

        return {"class": predicted_soil_class, "predictions": soil_predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during soil prediction: {e}")
@app.post("/predict/rice")
async def predict_paddy(file: UploadFile = File(...)):
    try:
        img = Image.open(BytesIO(await file.read()))
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict soil type
        soil_predictions = paddy_model.predict(img_array)
        predicted_soil_class = paddy_class_names[np.argmax(soil_predictions)]

        return {"class": predicted_soil_class, "predictions": soil_predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during soil prediction: {e}")
@app.post("/predict/banana")
async def predict_banana(file: UploadFile = File(...)):
    try:
        img = Image.open(BytesIO(await file.read()))
        img = img.resize((256, 256))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict soil type
        soil_predictions = banana_model.predict(img_array)
        predicted_soil_class = banana_class_names[np.argmax(soil_predictions)]

        return {"class": predicted_soil_class, "predictions": soil_predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during soil prediction: {e}")
@app.post("/predict/potato")
async def predict_potato(file: UploadFile = File(...)):
    try:
        img = Image.open(BytesIO(await file.read()))
        img = img.resize((256, 256))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict soil type
        soil_predictions = potato_model.predict(img_array)
        predicted_soil_class = potato_class_names[np.argmax(soil_predictions)]

        return {"class": predicted_soil_class, "predictions": soil_predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during soil prediction: {e}")
@app.post("/predict/sugarcane")
async def predict_sugarcane(file: UploadFile = File(...)):
    try:
        img = Image.open(BytesIO(await file.read()))
        img = img.resize((256, 256))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict soil type
        soil_predictions = sugarcane_model.predict(img_array)
        predicted_soil_class = sugarcane_class_names[np.argmax(soil_predictions)]

        return {"class": predicted_soil_class, "predictions": soil_predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during soil prediction: {e}")

@app.post("/predict/tomato")
async def predict_tomato(file: UploadFile = File(...)):
    try:
        img = Image.open(BytesIO(await file.read()))
        img = img.resize((256, 256))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict soil type
        soil_predictions = tomato_model.predict(img_array)
        predicted_soil_class = tomato_class_names[np.argmax(soil_predictions)]

        return {"class": predicted_soil_class, "predictions": soil_predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during soil prediction: {e}")







if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='localhost', port=8009)
