import sqlite3
from collections import Counter
from sqlite3 import Cursor

import nltk
from nltk.util import ngrams

nltk.download("punkt_tab")


def count_ngrams(cursor: Cursor):
    cursor.execute("SELECT message FROM turkish ORDER BY RANDOM() LIMIT 150000")

    unigrams = Counter()
    bigrams = Counter()
    trigrams = Counter()
    for row in cursor:
        tokens = nltk.word_tokenize(row[0], language="turkish")

        for token in tokens:
            unigrams[token] += 1
        for bigram in ngrams(tokens, 2):
            bigrams[bigram] += 1
        for trigram in ngrams(tokens, 3):
            trigrams[trigram] += 1

    print(f"Top 15 unigram: {unigrams.most_common(15)}")
    print(f"Top 15 bigram: {bigrams.most_common(15)}")
    print(f"Top 15 trigram: {trigrams.most_common(15)}")


def count_unique_users(cursor: Cursor):
    cursor.execute("SELECT COUNT(DISTINCT username) FROM turkish")
    users = cursor.fetchone()
    print(f"Total users: {users[0]}")

    cursor.execute("""SELECT username, COUNT(*) AS frequency
FROM turkish
GROUP BY username
ORDER BY frequency DESC;""")
    users_by_frequency = cursor.fetchall()
    print(f"Total users: {users_by_frequency[:10]}")


if __name__ == "__main__":

    con = sqlite3.connect("Turkish.db")
    cur = con.cursor()

    count_unique_users(cur)

