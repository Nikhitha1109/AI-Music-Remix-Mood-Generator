from flask import (
    Flask,
    render_template,
    request,
    send_from_directory
)

import os
import librosa
import numpy as np
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from datetime import datetime

from remix_generator import (
    generate_slow_remix,
    generate_fast_remix,
    generate_pitch_remix,
    generate_lofi_remix
)

from database import (
    db,
    Song
)

app = Flask(__name__)

UPLOAD_FOLDER = "../uploads"
REMIX_FOLDER = "../remixes"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["SQLALCHEMY_DATABASE_URI"] = \
    "sqlite:///music.db"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    REMIX_FOLDER,
    exist_ok=True
)

os.makedirs(
    "static",
    exist_ok=True
)

# Load Genre Model
model = joblib.load(
    "genre_model.pkl"
)


@app.route("/")
def home():
    return render_template(
        "index.html"
    )


@app.route(
    "/upload",
    methods=["POST"]
)
def upload_file():

    if "song" not in request.files:
        return "No file selected"

    file = request.files["song"]

    if file.filename == "":
        return "No file selected"

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    try:

        # Load Audio
        y, sr = librosa.load(
            filepath,
            sr=None
        )

        # Duration
        duration = librosa.get_duration(
            y=y,
            sr=sr
        )

        # Tempo
        tempo, beats = librosa.beat.beat_track(
            y=y,
            sr=sr
        )

        if isinstance(
            tempo,
            np.ndarray
        ):
            tempo = float(
                tempo[0]
            )
        else:
            tempo = float(
                tempo
            )

        # MFCC Features
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=13
        )

        feature = np.mean(
            mfcc.T,
            axis=0
        )

        feature = feature.reshape(
            1,
            -1
        )

        # Genre Prediction
        genre = model.predict(
            feature
        )[0]

        # Mood Detection
        if tempo > 120:
            mood = "Energetic"

        elif tempo > 90:
            mood = "Happy"

        elif tempo > 70:
            mood = "Relaxed"

        else:
            mood = "Sad"

        # Recommendation
        if mood == "Energetic":
            recommendation = \
                "Fast Remix"

        elif mood == "Happy":
            recommendation = \
                "Pitch Remix"

        elif mood == "Relaxed":
            recommendation = \
                "Lo-Fi Remix"

        else:
            recommendation = \
                "Slow Remix"

        # Generate Remixes
        generate_slow_remix(
            filepath
        )

        generate_fast_remix(
            filepath
        )

        generate_pitch_remix(
            filepath
        )

        generate_lofi_remix(
            filepath
        )

        # Save to Database
        new_song = Song(

            filename=file.filename,

            genre=genre,

            mood=mood,

            recommendation=recommendation,

            upload_date=datetime.now()

        )

        db.session.add(
            new_song
        )

        db.session.commit()

        return render_template(

            "result.html",

            filename=file.filename,

            duration=round(
                duration,
                2
            ),

            tempo=round(
                tempo,
                2
            ),

            genre=genre,

            mood=mood,

            recommendation=recommendation

        )

    except Exception as e:
        return f"Error: {str(e)}"


@app.route(
    "/download/<filename>"
)
def download_file(
    filename
):

    return send_from_directory(
        REMIX_FOLDER,
        filename,
        as_attachment=True
    )


@app.route(
    "/history"
)
def history():

    songs = Song.query.order_by(
        Song.id.desc()
    ).all()

    return render_template(
        "history.html",
        songs=songs
    )

@app.route("/dashboard")
def dashboard():

    songs = Song.query.all()

    total_songs = len(songs)

    genres = {}
    moods = {}

    for song in songs:

        genres[song.genre] = \
            genres.get(song.genre, 0) + 1

        moods[song.mood] = \
            moods.get(song.mood, 0) + 1

    os.makedirs("static", exist_ok=True)

    # Genre Chart

    if genres:

        plt.figure(figsize=(6, 6))

        plt.pie(
            list(genres.values()),
            labels=list(genres.keys()),
            autopct="%1.1f%%"
        )

        plt.title(
            "Genre Distribution"
        )

        plt.savefig(
            "static/genre_chart.png"
        )

        plt.close()

    # Mood Chart

    if moods:

        plt.figure(figsize=(6, 4))

        plt.bar(
            list(moods.keys()),
            list(moods.values())
        )

        plt.title(
            "Mood Distribution"
        )

        plt.savefig(
            "static/mood_chart.png"
        )

        plt.close()

    return render_template(
        "dashboard.html",
        total_songs=total_songs
    )
    
with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(
        debug=True
    )