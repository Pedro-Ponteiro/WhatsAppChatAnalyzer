import os
import re
from collections import Counter
from typing import List, Tuple

import nltk
import pandas as pd
from rake_nltk import Rake


def read_all_chat_files(folder_path, save: bool = False) -> List[str]:
    all_lines = ""
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
            all_lines += f.read().split("\n", maxsplit=1)[1]

    if save:
        with open("all_chat_data.txt", "w", encoding="utf-8") as f:
            f.write(all_lines)
    return all_lines


def get_messages_list(chat_lines: List[str]) -> List[List[str]]:

    re_compiler = re.compile(
        r"""(\d{1,2})/ 
            (\d{1,2})/
            (\d{1,2}),\s
            (\d{1,2}:
            \d{2}\s
            A?P?M)\s-\s
            (.*?):\s
            (.*?)
            (?=\d{1,2}/\d{1,2}/\d{1,2},\s)""",  # next message m/d/y
        re.VERBOSE | re.DOTALL,
    )
    matches = re_compiler.findall(chat_lines)

    return matches


def get_common_words(
    messages_col: pd.Series, sender: str, n: int = 100
) -> List[Tuple[str, str, int]]:
    messages_col = messages_col.copy()

    messages_col = messages_col.str.replace(r"\n$", "", regex=True)
    messages_col = messages_col.str.replace(r"[\.,]", "", regex=True)
    messages_col = messages_col.drop(
        messages_col.loc[messages_col.str.lower() == "<media omitted>"].index
    )

    results = messages_col.str.cat(sep=" ")
    results = re.split("\s+", results)

    STOPWORDS = nltk.corpus.stopwords.words("portuguese")
    results = [word for word in results if word.lower() not in STOPWORDS]

    results = [word.capitalize() for word in results]

    counter = Counter(results)

    return [(sender, word, count) for word, count in counter.most_common(n)]


def get_rake_classification(messages_col: pd.Series, n: int = 100):
    messages_col = messages_col.copy()

    messages_col = messages_col.drop(
        messages_col.loc[messages_col.str.lower() == "<media omitted>"].index
    )
    messages_col = messages_col.str.replace(r"\n$", ".\n", regex=True)

    results = messages_col.str.cat(sep="")

    rake = Rake()

    rake.extract_keywords_from_text(results)

    return rake.get_ranked_phrases()[:n]


def transform_to_df(formated_lines: List[Tuple[str, str, str]]) -> pd.DataFrame:
    df = pd.DataFrame(
        formated_lines,
        columns=[
            "month",
            "day",
            "year",
            "time",
            "sender",
            "message",
        ],
    )
    df = df.astype(
        dtype={
            "month": int,
            "day": int,
            "year": int,
            "time": str,
            "sender": str,
            "message": str,
        }
    )
    df["time"] = pd.to_datetime(df["time"], format="%I:%M %p").dt.time

    # retirar senders com menos de 10 mensagens (#TODO: CORRIGIR BUG QUE GERA ISSO)
    value_c = df["sender"].value_counts()
    menores_que_10 = value_c.loc[value_c < 10].index
    df = df.drop(df.loc[df["sender"].isin(menores_que_10)].index)

    return df


def get_word_count_data(df: pd.DataFrame, n: int = 100) -> List[Tuple[str, str, int]]:
    df = df.copy()
    word_count_data = []
    for sender in df["sender"].unique():
        sender_messages = df.loc[df["sender"] == sender, "message"]
        word_count_data.extend(get_common_words(sender_messages, sender, n))

    return word_count_data


def main() -> None:
    chat_lines = read_all_chat_files("./chats", save=True)
    chat_lines_formated = get_messages_list(chat_lines)
    df = transform_to_df(chat_lines_formated)

    df.to_excel("all_messages.xlsx", index=None)

    word_count_data = get_word_count_data(df, None)

    df_word_count = pd.DataFrame(
        word_count_data,
        columns=["sender", "message", "count"],
        dtype=str,
    )

    frases_chave = get_rake_classification(df["message"], None)
    frases_chave = [frase + "\n\n" for frase in frases_chave]

    df_word_count.to_excel("word_count_data.xlsx", index=None)
    with open("rake_phrases.txt", "w", encoding="utf-8") as f:
        f.writelines(frases_chave)


if __name__ == "__main__":
    main()
