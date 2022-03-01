import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import hstack
from sklearn.linear_model import SGDClassifier

def extract_to_bag():
    ratings=pd.read_table('data/train.tsv')
    x_all=ratings['Phrase']
    y_all=ratings['Sentiment']


if __name__=='__main__':
    extract_to_bag()


