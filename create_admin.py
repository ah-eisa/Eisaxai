#!/usr/bin/env python3
"""
Run once to create the first admin user.
Usage:
    python3 create_admin.py --email admin@eisax.com --name "Ahmed Eisa" [--password MyPass123]
"""
import sys, argparse
sys.path.insert(0, '/home/ubuntu/investwise')

from core.user_db import init_users_table, create_user, get_user_by_email
from core.auth import hash_password, generate_temp_password

parser = argparse.ArgumentParser()
parser.add_argument('--email',    required=True)
parser.add_argument('--name',     required=True)
parser.add_argument('--password', default=None, help='Leave blank to auto-generate')
args = parser.parse_args()

init_users_table()

if get_user_by_email(args.email):
    print(f"[!] User {args.email} already exists.")
    sys.exit(0)

pw = args.password or generate_temp_password()
uid = create_user(
    email=args.email,
    name=args.name,
    password_hash=hash_password(pw),
    role='admin',
    must_change_pw=(args.password is None),
)

print(f"\n✓ Admin created")
print(f"  ID      : {uid}")
print(f"  Email   : {args.email}")
print(f"  Name    : {args.name}")
print(f"  Password: {pw}")
if args.password is None:
    print(f"  → Must change password on first login")
