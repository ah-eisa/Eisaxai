from core.database import DatabaseManager
import os
db = DatabaseManager("test_invest.db")
db.create_session("test_1", "ahmed_eisa")
db.add_message("test_1", "user", "Hello EisaX")
history = db.get_chat_history("test_1")
if len(history) > 0:
    print("✅ Database System is Up and Running!")
    os.remove("test_invest.db")
