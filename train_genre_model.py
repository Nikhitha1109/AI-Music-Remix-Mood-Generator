import os
import librosa
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATASET_PATH = "../datasets/genres_original"

X = []
y = []

for genre in os.listdir(DATASET_PATH):

    genre_folder = os.path.join(
        DATASET_PATH,
        genre
    )

    if not os.path.isdir(genre_folder):
        continue

    print("Processing:", genre)

    for song in os.listdir(genre_folder):

        file_path = os.path.join(
            genre_folder,
            song
        )

        try:

            audio, sr = librosa.load(
                file_path,
                duration=30
            )

            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=13
            )

            feature = np.mean(
                mfcc.T,
                axis=0
            )

            X.append(feature)
            y.append(genre)

        except Exception as e:
            print("Skipped:", song)

X = np.array(X)
y = np.array(y)

print("Total Samples:", len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("Accuracy:", accuracy)

joblib.dump(
    model,
    "genre_model.pkl"
)

print("Model Saved Successfully")