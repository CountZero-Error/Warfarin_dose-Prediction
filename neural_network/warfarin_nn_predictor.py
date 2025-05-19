import torch
import joblib
import numpy as np
from torch import nn

"""================================== NEURAL NETWORK =================================="""
# Feed Forward Neural Network
class WarfarinNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.NN = nn.Sequential(
            nn.Linear(20, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.LeakyReLU(),

            nn.Linear(64, 1) # output layer – single continuous value (mg/week)
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        # Kaiming-uniform initialisation suited for LeakyReLU.
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.NN(x)

def predict_warfarin_dose(input_features):
    # Load the model
    model = WarfarinNN()
    model.load_state_dict(torch.load("best_check_point/best_nn.pt"))
    model.eval()

    # Load scalers
    X_scaler = joblib.load("best_check_point/Xscaler.pkl")
    y_scaler = joblib.load("best_check_point/yscaler.pkl")

    """
    input_features: list or numpy array of shape (20,)
    Returns: predicted warfarin dose (float)
    """
    input_scaled = X_scaler.transform([input_features])
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor).numpy()

    dose = y_scaler.inverse_transform(prediction)[0][0]

    return dose


if __name__ == "__main__":
    example_input = [2.13,75.5,173.48199999981264,1,84.5,1,0,0,0,0,0,0,0,0.0,0.0,0.0,0.0,0.0,0.0,25.08638661377243]  # 35.0 mg/week
    predicted_dose = predict_warfarin_dose(example_input)
    print(f"Predicted Warfarin Dose: {predicted_dose:.2f} mg/week")