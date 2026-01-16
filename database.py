import sqlite3

conn = sqlite3.connect("lung.db", check_same_thread=False)
cursor = conn.cursor()

# USERS
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT UNIQUE,
    password TEXT,
    role TEXT
)
""")

# REPORTS
cursor.execute("""
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER,
    prediction TEXT,
    benign REAL,
    malignant REAL,
    pdf_path TEXT,
    doctor_remark TEXT
)
""")

conn.commit()

# ---------- USERS ----------
def create_user(name, email, password, role):
    cursor.execute(
        "INSERT INTO users VALUES (NULL,?,?,?,?)",
        (name, email, password, role)
    )
    conn.commit()

def get_user_by_email(email):
    cursor.execute("SELECT * FROM users WHERE email=?", (email,))
    return cursor.fetchone()

def delete_user(user_id):
    cursor.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()

def get_all_users():
    cursor.execute("SELECT id,name,email,role FROM users")
    return cursor.fetchall()

# ---------- REPORTS ----------
def save_report(patient_id, prediction, benign, malignant, pdf_path):
    cursor.execute(
        "INSERT INTO reports VALUES (NULL,?,?,?,?,?,NULL)",
        (patient_id, prediction, benign, malignant, pdf_path)
    )
    conn.commit()

def get_patient_reports(patient_id):
    cursor.execute(
        "SELECT * FROM reports WHERE patient_id=?",
        (patient_id,)
    )
    return cursor.fetchall()

def get_all_reports():
    cursor.execute("SELECT * FROM reports")
    return cursor.fetchall()

def add_doctor_remark(report_id, remark):
    cursor.execute(
        "UPDATE reports SET doctor_remark=? WHERE id=?",
        (remark, report_id)
    )
    conn.commit()
