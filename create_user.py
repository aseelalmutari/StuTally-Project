# create_user.py
import sqlite3
from flask_bcrypt import Bcrypt

# configure Bcrypt
bcrypt = Bcrypt()

# hash your desired password
password_plain = "admin123"
pw_hash = bcrypt.generate_password_hash(password_plain).decode('utf-8')

conn = sqlite3.connect('data/detections.db')
conn.execute("""
  CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT UNIQUE,
    password TEXT,
    role TEXT
  )
""")
conn.execute(
  "INSERT OR IGNORE INTO users (username, password, role) VALUES (?,?,?)",
  ("admin", pw_hash, "admin")
)
conn.commit()
conn.close()

print("User table ensured and admin user inserted (if not existing).")
