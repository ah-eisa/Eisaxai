import json

from core.user_db import list_users


def main():
    users = list_users()
    out = []
    for u in users:
        out.append(
            {
                "id": u.get("id"),
                "email": u.get("email"),
                "role": u.get("role"),
                "is_active": u.get("is_active"),
                "must_change_pw": u.get("must_change_pw"),
            }
        )
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
