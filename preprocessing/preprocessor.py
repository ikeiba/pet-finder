#imports 

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder 


#We first define the columns groups. Each group will be applied a different preprocessing 
#technique (scaling, one-hot encoding or target-encoding)


numeric_cols = ['Age', 'Fee', 'PhotoAmt', 'VideoAmt', 'Image_Brightness', 'Subject_Focus_Ratio', "Crop_Confidence", "Visual_Puppy_Score", "NLP_Sentiment_Score", "NLP_Emotional_Intensity", "Name_length"]
onehot_enc_cols = ['Type', 'Gender', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'RescuerID']
target_enc_cols = ['Breed1', 'Breed2', 'State']

preprocessor = ColumnTransformer (transformers = [
    ("standard_scaler", StandardScaler(), numeric_cols),
    ("one_hot_enc", OneHotEncoder(handle_unknown="ignore"), onehot_enc_cols),
    ("target_enc",TargetEncoder(), target_enc_cols)
], remainder='passthrough')


#For original data we only need the columns that were in the original csv
orig_numeric_cols = ['Age', 'Fee', 'PhotoAmt', 'VideoAmt', 'Quantity']
orig_onehot_enc_cols = ['Type', 'Gender', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'RescuerID', 'Color1', 'Color2', 'Color3', 'MaturitySize', 'FurLength']
orig_target_enc_cols = ['Breed1', 'Breed2', 'State']


originaldf_preprocessor = ColumnTransformer (transformers = [
    ("standard_scaler", StandardScaler(), orig_numeric_cols),
    ("one_hot_enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False), orig_onehot_enc_cols),
    ("target_enc",TargetEncoder(), orig_target_enc_cols)
], remainder='passthrough')



"""Example of Usage in a Pipeline with Optuna and cros validation 

def objective(trial):
    # Sugerimos hiperparámetros para el modelo (ej: RandomForest)
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)

    # Creamos el Pipeline completo
    # El preprocesador va PRIMERO, luego el clasificador
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', clf)
    ])

    # Evaluamos con Cross-Validation (esto evita el leakage de forma automática)
    # TargetEncoder solo verá el 'y' del fold de entrenamiento en cada vuelta
    score = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy').mean()
    
    return score

# 4. Lanzamos la optimización
# (Asumiendo que ya tienes X_train e y_train)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print(f"Mejores parámetros: {study.best_params}")


"""