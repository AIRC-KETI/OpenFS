import pandas as pd
import re

def save_words(words, data_type):
    words_uniq = list(set(words))
    with open(f'data/english_word/neobench_{data_type}.txt', 'w') as f:
        for word in words_uniq:
            if re.fullmatch(r'[A-Za-z]+', word):
                f.write(f'{word}\n')


if __name__ == "__main__":
    df = pd.read_excel('data/english_word/Neologisms.xlsx')
    test_words = df['Neologism']
    save_words(test_words, data_type='test')