import json
import secrets
import string
import sys

from core.auth import hash_password
from core.user_db import get_user_by_email, update_user


def _gen_password(length: int = 20) -> str:
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()-_=+"
    while True:
        pw = "".join(secrets.choice(alphabet) for _ in range(length))
        if (
            any(c.islower() for c in pw)
            and any(c.isupper() for c in pw)
            and any(c.isdigit() for c in pw)
            and any(c in "!@#$%^&*()-_=+" for c in pw)
        ):
            return pw


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "email argument required"}))
        raise SystemExit(1)

    email = sys.argv[1].strip().lower()
    user = get_user_by_email(email)
    if not user:
        print(json.dumps({"ok": False, "error": f"user not found: {email}"}))
        raise SystemExit(2)

    temp_pw = _gen_password()
    ok = update_user(
        int(user["id"]),
        password_hash=hash_password(temp_pw),
        must_change_pw=1,
        is_active=1,
    )
    print(
        json.dumps(
            {
                "ok": bool(ok),
                "email": user["email"],
                "role": user["role"],
                "must_change_pw": True,
                "temp_password": temp_pw,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
