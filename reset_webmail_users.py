import json
import secrets
import string

from core.auth import hash_password
from core.user_db import get_user_by_email, list_users, update_user


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


def _find_admin_user():
    users = list_users()
    for u in users:
        if (u.get("role") or "").lower() == "admin":
            return u
    for u in users:
        if "admin" in (u.get("email") or "").lower():
            return u
    return None


def _reset_user(user: dict, temp_password: str):
    return update_user(
        int(user["id"]),
        password_hash=hash_password(temp_password),
        must_change_pw=1,
        is_active=1,
    )


def main():
    result = {"ok": False, "resets": [], "errors": []}

    ahmed = get_user_by_email("ahmed@eisax.com")
    if not ahmed:
        result["errors"].append("User not found: ahmed@eisax.com")
    else:
        pw = _gen_password()
        if _reset_user(ahmed, pw):
            result["resets"].append(
                {
                    "email": ahmed["email"],
                    "role": ahmed["role"],
                    "must_change_pw": True,
                    "temp_password": pw,
                }
            )
        else:
            result["errors"].append("Failed to reset: ahmed@eisax.com")

    admin = _find_admin_user()
    if not admin:
        result["errors"].append("Admin user not found")
    else:
        pw = _gen_password()
        if _reset_user(admin, pw):
            result["resets"].append(
                {
                    "email": admin["email"],
                    "role": admin["role"],
                    "must_change_pw": True,
                    "temp_password": pw,
                }
            )
        else:
            result["errors"].append(f"Failed to reset admin: {admin.get('email')}")

    result["ok"] = len(result["errors"]) == 0
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
