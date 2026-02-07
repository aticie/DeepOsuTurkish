import datetime
from zipfile import ZipFile
import sqlite3

from tqdm import tqdm

if __name__ == '__main__':
    con = sqlite3.connect("Chatlogs.db")
    cursor = con.cursor()

    tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()

    with ZipFile('chatlogs.zip', 'w') as chat_zip:
        for table_tup in tqdm(tables):
            table = table_tup[0]
            if table == "sqlite_sequence":
                continue

            old_date = "2000-01-01"
            file_name = f"temp.log"
            batched_text = []
            all_lines = cursor.execute(f"SELECT * FROM {table}").fetchall()
            for _, hour, username, message, date in tqdm(all_lines):

                if date != old_date:
                    old_dt = datetime.datetime.strptime(old_date, "%Y-%m-%d")
                    batched_text.append(old_dt.strftime("--- Log closed %a %b %d 00:00:00 %Y"))
                    old_date = date
                    chat_zip.writestr(file_name, "\n".join(batched_text))
                    file_name = f"#{table}/{date}.log"
                    dt = datetime.datetime.strptime(date, "%Y-%m-%d")
                    batched_text = [dt.strftime("--- Log opened %a %b %d 00:00:00 %Y")]

                if len(message) and message[0] == "\x01":
                    formatted_msg = f"{hour}:00  * {username} {message[8:-1]}"
                else:
                    formatted_msg = f"{hour}:00 < {username}> {message}"
                batched_text.append(formatted_msg)

