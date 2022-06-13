# Importing Libraries

import pandas as pd
import joblib

# Getting your sample data
# Here you would use your own CSV file with your sample data, for this example we used the file named: "askvill_sample_HE_TRI_aSMA_CD163.csv"

ValData=pd.read_csv('askvill_sample_HE_TRI_aSMA_CD163.csv',index_col=None, encoding = 'unicode_escape' )
ValData.columns = ValData.columns.str.replace('[#,@,&, ,%, ², µ]', '_')
X_val, y_val = ValData.iloc[:,3:],ValData["Ashcroft_(Average)"]

# Loading askvill model from file and making predictions
# Here you would use your preferred model, for this example we used the model named "Askvill_Score_HE_TRI_aSMA_CD163.sav" but you could also choose the other model

loaded_model = joblib.load("Askvill_Score_HE_TRI_aSMA_CD163.sav")
result = loaded_model.score(X_val, y_val)
preds = loaded_model.predict(X_val)

# Printing results

print("r2 score = ", result)
target = pd.DataFrame(y_val)
predictions = pd.DataFrame(preds, columns=['Model Prediction'])
table = pd.concat([target, predictions], axis=1)
table
