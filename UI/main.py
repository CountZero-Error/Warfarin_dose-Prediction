from PyQt5.QtWidgets import QApplication, QMainWindow
from functools import partial
import window
import pickle
import sys

def prediction(ui, model):
    # "Gender", "Age", "Height (cm)", "Weight (kg)", "INR on Reported Therapeutic Dose of Warfarin", "Current Smoker"
    Gender = {'female': 0, 'male': 1}
    Smoker = {'Yes': 1, "No": 0}
    Age = {'10-19': 0, '20-29': 1, '30-39': 2, '40-49': 3, '50-59': 4, '60-69': 5, '70-79': 6, '80-89': 7, '90': 8}

    gender = ui.gender.currentText()
    gender_code = Gender[gender]

    age = int(ui.age.text())
    age_code = 0
    for k, v in Age.items():
        if age >= 90:
            age_code = 8
        else:
            min_max = k.split('-')
            if int(min_max[0]) <= age <= int(min_max[1]):
                age_code = v
    
    smoker = ui.smoker.currentText()
    smoker_code = Smoker[smoker]

    height = ui.height.text()
    weight = ui.weight.text()
    INR = ui.INR.text()

    data = [[float(gender_code), float(age_code), float(height), float(weight), float(INR), float(smoker_code)]]
    print(f'data:{data}')

    if data[0][-2] == 0.:
        ui.textBrowser.setText(f'No need to take warfarin.')
    else:
        dose = model.predict(data)
        ui.textBrowser.setText(f'The dose of warfarin your should take is {str(round(dose[0]/6))} mg/day.')
        print(f'result:{round(dose[0]/6)}\n')

if __name__ == '__main__':
    # Load model
    with open('Warfarin_dose-Prediction-main\models\RFR.pkl', 'rb') as filo:
        model = pickle.load(filo)

    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = window.Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    ui.pushButton.clicked.connect(partial(prediction, ui, model))
    sys.exit(app.exec_())
