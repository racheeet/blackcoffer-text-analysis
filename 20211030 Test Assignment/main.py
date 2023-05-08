import chardet
import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
nltk.download('stopwords')
nltk.download('punkt')

directory = '/content/drive/MyDrive/test_assignment/20211030 Test Assignment-20230419T154448Z-001/20211030 Test Assignment'


path_to = {
    'auditor': '{}/StopWords/StopWords_Auditor.txt'.format(directory),
    'dates_number': '{}/StopWords/StopWords_DatesandNumbers.txt'.format(directory),
    'generic': '{}/StopWords/StopWords_Generic.txt'.format(directory),
    'generic_long': '{}/StopWords/StopWords_GenericLong.txt'.format(directory),
    'geographic':'{}/StopWords/StopWords_Geographic.txt'.format(directory),
    'names': '{}/StopWords/StopWords_Names.txt'.format(directory),
    'currencies': '{}/StopWords/StopWords_Currencies.txt'.format(directory),
    'negative': '{}/MasterDictionary/negative-words.txt'.format(directory),
    'positive': '{}/MasterDictionary/positive-words.txt'.format(directory),
    'input_data': '{}/Input.xlsx'.format(directory)
}

# @title helper functions
def get(url):
    try:
        response = requests.get(url)
    except Exception as e:
        print(e)

    soup = BeautifulSoup(response.text, 'html.parser')
    try:
        title = soup.title.text.strip()
        title = title.split('|')[0]
    except Exception as e:
        try:
            title = soup.find("h1", {'class': "tdb-title-text"}).text.strip()
        except:
            None

    text = ""
    content = soup.find('div', {'class': 'td-post-content'})
    if content is None:
        return None, None

    for p in content.find_all('p'):
        for strong in p.find_all('strong'):
            strong.decompose()  # Remove <strong> tags and their contents
        text += p.get_text().strip()

    return title, text

def get_titles_texts():
    titles = []
    texts = []
    for url in df_content.URL:
        title, text = get(url)
        titles.append(title)
        texts.append(text)
    return titles, texts

def get_stopwords_from(aud_path = path_to['auditor'], dn_path=path_to['dates_number'], genr_path=path_to['generic'], genrL_path=path_to['generic_long'],geo_path = path_to['geographic'], names_path = path_to['names'],currnc_path=path_to['currencies']):
    with open(aud_path, 'rb') as f:
        result = chardet.detect(f.read())
    with open(aud_path, "r", encoding=result['encoding']) as file:
        lines = file.readlines()
        auditor = [line.strip() for line in lines]

    with open(dn_path, 'rb') as f:
        result = chardet.detect(f.read())
    with open(dn_path, "r", encoding=result['encoding']) as file:
        lines = file.readlines()
        date_numbers = [line.strip() for line in lines]

    with open(genr_path, 'rb') as f:
        result = chardet.detect(f.read())
    with open(genr_path, "r", encoding=result['encoding']) as file:
        lines = file.readlines()
        generic = [line.strip() for line in lines]

    with open(genrL_path, 'rb') as f:
        result = chardet.detect(f.read())
    with open(genrL_path, "r", encoding=result['encoding']) as file:
        lines = file.readlines()
        generic_long = [line.strip() for line in lines]

    with open(geo_path, 'rb') as f:
        result = chardet.detect(f.read())
    with open(geo_path, "r", encoding=result['encoding']) as file:
        lines = file.readlines()
        geographic = [line.strip() for line in lines]

    with open(names_path, 'rb') as f:
        result = chardet.detect(f.read())
    with open(names_path, "r", encoding=result['encoding']) as file:
        lines = file.readlines()
        names = [line.strip() for line in lines]

    with open(currnc_path, 'rb') as f:
        result = chardet.detect(f.read())
    with open(currnc_path, 'r', encoding=result['encoding']) as f:
        lines = f.readlines()
        currencies = [line.strip() for line in lines]

    currencies_modified = [x.split('|') for x in currencies]
    currencies_redundant = []
    for words in currencies_modified:
        for word in words:
            currencies_redundant.append(word.strip())
            
    txt_files_stopwords = set(auditor + currencies + geographic + names + generic+generic_long + date_numbers + currencies_redundant)
    
    stop_words = set(stopwords.words('english')).union(txt_files_stopwords)
    for words in stop_words:
        words = words.lower()
        
    return stop_words

def filter_sentence(text,stop_words):
  if not isinstance(text, str):
    return ''
  sentence = text
  word_tokens = word_tokenize(sentence)
  filtered_sentence = " ".join([w for w in word_tokens if w not in stop_words])
  return filtered_sentence

def get_categorical_words(filename_pos = path_to['positive'], filename_nega = path_to['negative']):
 
    with open(filename_pos, 'rb') as f:
        result = chardet.detect(f.read())
    with open(filename_pos, "r") as file:
        lines = file.readlines()
        positive_words = [line.strip() for line in lines]

    with open(filename_nega, 'rb') as f:
        result = chardet.detect(f.read())
    with open(filename_nega, 'r', encoding=result['encoding']) as f:
        lines = f.readlines()
        negative_words = [line.strip() for line in lines]
        
    return positive_words, negative_words

def get_score(filtered_sentence, positive_words, negative_words):
    if not isinstance(filtered_sentence, str):
        return ''
    negative_score = 0
    positive_score = 0
    temp = filtered_sentence.split(' ')
    for word in temp:
        if word in positive_words:
            positive_score += 1
        elif word in negative_words:
            negative_score += -1
    return positive_score, -1*negative_score

def get_polarity_scores(df,col1 = 'POSITIVE SCORE', col2 = 'NEGATIVE SCORE'):
    polarity_scores = []
    for pos_score, neg_score in zip(df[col1], df[col2]):
        polarity_score = (pos_score - neg_score)/(pos_score + neg_score + 0.000001)
        polarity_scores.append(polarity_score)
    return polarity_scores

def get_words_and_sentences(df, col1 = 'filtered_texts'):
    words = []
    sent = []
    avg_sent_length = []

    for sentences in df[col1]:
        sent_count = word_count = 0 
        for w in sentences.split(" "):
            if w in ['.','?','!']:
                sent_count +=1
            elif w not in ['%', '(', ')', ',', '’', '“', '”']:
                word_count += 1
        words.append(word_count)
        sent.append(sent_count)
        avg_sent_length.append(word_count/(sent_count + 0.000001))

    return avg_sent_length, words, sent

def get_subjectivity_scores(df, col1='POSITIVE SCORE', col2='NEGATIVE SCORE', col3='WORD COUNT'):
    sub_scores = []
    for pos, neg, words in zip(df[col1],df[col2],df[col3]):
        sub_score = (pos + neg)/ (words + 0.000001)
        sub_scores.append(sub_score)
    return sub_scores

def count_complex_words(paragraph):
    vowels = re.compile(r'[aeiouyAEIOUY]+')
    exceptions = set(['es', 'ed'])
    words = paragraph.split()
    complex_syllables = 0
    syllables = 0
    for word in words:
        word = word.rstrip('.,;:?!')
        num_vowels = len(vowels.findall(word))

        if num_vowels > 0 and word[-2:] not in exceptions:
            syllables += 1   
        if num_vowels > 2 and word[-2:] not in exceptions:
            complex_syllables += 1
   
    return complex_syllables, syllables

def count_personal_pronouns(text):
    personal_pronouns = ["I", "we", "my", "ours", "us"]
    pattern = r"\b(" + "|".join(personal_pronouns) + r")\b"
    matches = re.findall(pattern, text, re.IGNORECASE)
    return len(matches)

def get_average_words(df, col = 'filtered_texts', col2= 'WORD COUNT'):
    avg_words_length = []
    for sentences, words in zip(df[col], df[col2]):
        chars = sentences.replace(" ", "").replace('.',"").replace("!","").replace("?","")
        awl = chars.__len__()/words
        avg_words_length.append(awl)
    return avg_words_length

def get_words_per_sentences(text):
    if not isinstance(text, str):
        return ''
    
    word_tokens = word_tokenize(text)
    s = " ".join([w for w in word_tokens])
    sent_count = word_count = 0
    for w in s.split(" "):
        if w in ['.','?','!']:
            sent_count +=1
        elif w not in ['%', '(', ')', ',', '’', '“', '”']:
            word_count += 1
    return (word_count/(sent_count + 0.000001))

def save_dataframe(df, path):
    df.to_excel('{}.xlsx'.format(path), index=False)
    df.to_csv('{}.csv'.format(path), index=False)   


# @title analyse
def analysed_texts_df(df):
    
    positive_words, negative_words = get_categorical_words()
    stop_words = get_stopwords_from()
    
    df['filtered_texts'] =  df['texts'].apply(filter_sentence, stop_words = stop_words)
    df['POSITIVE SCORE'], df['NEGATIVE SCORE'] = zip(*df.filtered_texts.apply(get_score,positive_words = positive_words, negative_words= negative_words))
    df['POLARITY SCORE'] = get_polarity_scores(df)
    df['AVG SENTENCE LENGTH'], df['WORD COUNT'], df['total_sentences'] = get_words_and_sentences(df)
    df['SUBJECTIVITY SCORE'] = get_subjectivity_scores(df)
    df['AVG NUMBER OF WORDS PER SENTENCE'] = (df.texts).apply(get_words_per_sentences)
    df['COMPLEX WORD COUNT'], df['SYLLABLES'] = zip(*df.filtered_texts.apply(count_complex_words))
    df['PERCENTAGE OF COMPLEX WORDS'] = df.apply(lambda row: 100*row['COMPLEX WORD COUNT']/row['WORD COUNT'], axis = 1)
    df['FOG INDEX'] = df.apply(lambda row: 0.4*(row['AVG SENTENCE LENGTH']) + (row['PERCENTAGE OF COMPLEX WORDS']), axis=1)
    df['SYLLABLE PER WORD'] = df.apply(lambda row: row['SYLLABLES']/(row['WORD COUNT']+ 0.000001), axis=1)
    df['PERSONAL PRONOUNS'] = df.filtered_texts.apply(count_personal_pronouns)
    df['AVG WORD LENGTH'] = get_average_words(df)

    col = 'AVG NUMBER OF WORDS PER SENTENCE'
    df.loc[df[col] == 1000000.0] = 'NaN'
    col = 'FOG INDEX'
    df.loc[df[col] == 4000000.0] = 'NaN'

    new_order = ['URL_ID','URL','POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE','AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT', 'WORD COUNT','SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH']
    df = df.reindex(columns=new_order)

    return df


# @title main
if __name__ == '__main__':
    
    df_content = pd.read_excel(path_to['input_data'])
    df_content['titles'], df_content['texts'] = get_titles_texts()
    
    # save 
    save_loc = 'scrapped_data'
    save_dataframe(df_content, save_loc)
    
    
    df = pd.read_csv('{}/{}.csv'.format(directory,save_loc))
    
    df = analysed_texts_df(df)
        
    # save
    save_loc = 'OUTPUT DATA STRUCTURE'
    save_dataframe(df, save_loc)