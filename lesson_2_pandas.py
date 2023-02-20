
import numpy as np

#Задание 1

import pandas as pd

authors = pd.DataFrame({'author_id': [1, 2, 3], 'author_name': ['Тургенев', 'Чехов', 'Островский']}, columns=['author_id', 'author_name'])

print("\nauthors:\n")
print(authors)

book = pd.DataFrame({'author_id': [1, 1, 1, 2, 2, 3, 3],
                     'book_title': ['Отцы и дети', 'Рудин', 'Дворянское гнездо',
                                    'Толстый и тонкий', 'Дама с собачкой',
                                    'Гроза','Таланты и поклонники'],
                     'price': [25, 31, 43, 20, 16, 104, 19]},
                    columns=['author_id', 'book_title', 'price'])

print("\nbook:\n")
print(book)

#Задание 2

authors_price = pd.merge(authors, book, on='author_id', how='outer')

print("\nauthors_price:\n")
print(authors_price)

#Задание 3

top5 = authors_price.nlargest(5, 'price')

print("\ntop5:\n")
print(top5)

#Задание 4

authors_stat = authors_price.groupby('author_name').agg({'price': ['min', 'max', 'mean']})
authors_stat = authors_stat.rename(columns={'min': 'min_price', 'max': 'max_price', 'mean': 'mean_price'})

print("\nauthors_stat:\n")
print(authors_stat)

#Задание 5**

authors_price['cover'] = ['твердая', 'мягкая', 'мягкая', 'твердая', 'твердая', 'мягкая', 'мягкая']

book_info = pd.pivot_table(authors_price, values='price', index=['author_name'], columns=['cover'], aggfunc=np.sum)
book_info['мягкая'] = book_info['мягкая'].fillna(0)
book_info['твердая'] = book_info['твердая'].fillna(0)

book_info.to_pickle('book_info.pkl')
book_info2 = pd.read_pickle('book_info.pkl')

var_eq = book_info.equals(book_info2)

print("\nauthors_price:\n")
print(authors_price)

print("\nbook_info:\n")
print(book_info)

print("\nis book_info = book_info2:\n")
print(var_eq)
