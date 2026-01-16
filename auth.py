import bcrypt
from database import create_user, get_user_by_email

# -------------------------
# REGISTER USER
# -------------------------
def register(name, email, password, role):
    hashed_password = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")   # store as STRING

    create_user(name, email, hashed_password, role)

# -------------------------
# LOGIN USER
# -------------------------
def login(email, password):
    user = get_user_by_email(email)
    if not user:
        return None

    stored_hash = user[3]  # password column (STRING)

    if bcrypt.checkpw(
        password.encode("utf-8"),
        stored_hash.encode("utf-8")
    ):
        return user

    return None
