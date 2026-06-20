import sqlite3

conn = sqlite3.connect(r'C:\Users\HYPERPC\PycharmProjects\tg\copy paste\bot\data\intraday.db')

tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("Таблицы:", tables)

for (t,) in tables:
    print(f"\n--- {t} ---")
    cols = conn.execute(f"PRAGMA table_info({t})").fetchall()
    for c in cols:
        print(f"  {c[1]} ({c[2]})")
    count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"  Записей: {count}")
    if count > 0:
        first = conn.execute(f"SELECT * FROM {t} LIMIT 1").fetchone()
        print(f"  Пример: {first}")

conn.close()
