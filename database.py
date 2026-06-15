from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Song(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    filename = db.Column(
        db.String(200),
        nullable=False
    )

    genre = db.Column(
        db.String(100),
        nullable=False
    )

    mood = db.Column(
        db.String(100),
        nullable=False
    )

    recommendation = db.Column(
        db.String(100),
        nullable=False
    )

    upload_date = db.Column(
        db.DateTime,
        nullable=False
    )

    def __repr__(self):

        return f"<Song {self.filename}>"