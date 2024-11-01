import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
from custom_gmm import CustomGMM


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'CustomGMM':
            from custom_gmm import CustomGMM
            return CustomGMM
        return super().find_class(module, name)

current_model = 'gmm_model.pkl'
gmm = CustomUnpickler(open('./' + current_model, 'rb')).load()

class PointRequest(BaseModel):
    x: float
    y: float


app = FastAPI()

@app.post("/predict/")
async def predict_probabilities(point: PointRequest):
    input_point = np.array([point.x, point.y])
    
    probabilities = gmm.predict_proba(input_point)
    probabilities = np.round(probabilities, decimals=5) 
        
    response = {
        "cluster_probabilities": probabilities.tolist()
    }
    
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    # for running: uvicorn gmm_fastapi:app --reload
