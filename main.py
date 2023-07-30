#Import necessary modules
import numpy as np
import joblib
import warnings

#prediction class
class ModelPredictions:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        self.model = joblib.load(self.model_path)

    def predict_values(self, arr):
        if self.model is None:
            self.load_model()

        arr = np.array(arr)
        arr = arr.reshape(1, -1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred_vals = self.model.predict(arr)

        predictions = {
            'MRR': pred_vals[0][0] if pred_vals[0][0] > 0 else "Negative value encountered",
            'TWR': pred_vals[0][1] if pred_vals[0][1] > 0 else "Negative value encountered",
            'Residual Stress': pred_vals[0][2] if pred_vals[0][2] > 0 else "Negative valueencountered"
        }

        return predictions


#driver
if __name__ == "__main__":
    model_path = 'model/modelWeightsSolverLBFGS.pkl'
    modelObj = ModelPredictions(model_path)

    #arr = [12.5, 35, 250, 75] #output = 69.547, 0, 7.54
    #arr = [17.5, 45, 150, 85] #output = 94.214, 4.214, 8.96
    #arr = [10, 50, 300, 70]  #output = 63.245, 4.214, 8.96
    arr = []
    arr.append(float(input("Enter Current: ")))
    arr.append(float(input("Enter Voltage: ")))
    arr.append(float(input("Enter Pulse On Time: ")))
    arr.append(float(input("Enter Duty On Factor: ")))

    predictions = modelObj.predict_values(arr)
    print(f"Predictions: {predictions}")
