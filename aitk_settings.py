"""
Renseigne automatiquement le token Hugging Face dans les Settings de l'interface ai-toolkit,
a partir de la variable HF_TOKEN du template RunPod.

Attend que la base de l'interface soit creee (au demarrage du pod), puis ecrit la valeur.
Lance en arriere-plan par la commande de demarrage du template.
"""
import os
import sqlite3
import sys
import time

DB_CANDIDATES = ["/app/ai-toolkit/aitk_db.db", "/workspace/ai-toolkit/aitk_db.db"]
VALUES = {"HF_TOKEN": os.environ.get("HF_TOKEN", "").strip()}
TIMEOUT = 600  # secondes


def find_table(con):
    for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        cols = {r[1].lower(): r for r in con.execute(f'PRAGMA table_info("{name}")')}
        if "setting" in name.lower() and "key" in cols and "value" in cols:
            return name, cols
    return None, None


def upsert(con, table, cols, key, value):
    kcol = cols["key"][1]
    vcol = cols["value"][1]
    row = con.execute(f'SELECT "{vcol}" FROM "{table}" WHERE "{kcol}"=?', (key,)).fetchone()
    if row is not None:
        if row[0] == value:
            return "deja a jour"
        con.execute(f'UPDATE "{table}" SET "{vcol}"=? WHERE "{kcol}"=?', (value, key))
        return "mis a jour"
    data = {kcol: key, vcol: value}
    now_ms = int(time.time() * 1000)
    for name, (_, real, typ, notnull, default, pk) in cols.items():
        if real in data or pk or not notnull or default is not None:
            continue
        data[real] = now_ms  # colonnes obligatoires sans valeur par defaut (dates)
    names = ", ".join(f'"{c}"' for c in data)
    marks = ", ".join("?" for _ in data)
    con.execute(f'INSERT INTO "{table}" ({names}) VALUES ({marks})', list(data.values()))
    return "ajoute"


def main():
    if not VALUES["HF_TOKEN"]:
        print("HF_TOKEN absent de l'environnement, rien a faire.")
        return 0
    start = time.time()
    while time.time() - start < TIMEOUT:
        for db in DB_CANDIDATES:
            if not os.path.exists(db):
                continue
            try:
                con = sqlite3.connect(db, timeout=10)
                table, cols = find_table(con)
                if table:
                    for k, v in VALUES.items():
                        print(f"{k} : {upsert(con, table, cols, k, v)} ({db}, table {table})")
                    con.commit()
                    con.close()
                    return 0
                con.close()
            except sqlite3.Error as e:
                print(f"attente ({e})")
        time.sleep(5)
    print("Base de l'interface introuvable apres 10 min.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
