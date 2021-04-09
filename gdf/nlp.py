import nltk
import yake
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# Download the corpus slicing utility
nltk.download('punkt')


def word_count(corpus) -> int:
    """Returns the number of words in a string.
    Args:
        corpus: The character string for which we want the number of words.
    Returns:
        The number of words in the `corpus` string
    """

    # Retrieving the list of words from the corpus
    words = tokenize(corpus)
    return len(words)


def occurrences(corpus) -> dict:
    """Perform a word mapping -> number of occurrences of the word.
    Args:
        corpus:
    The corpus on which we want to perform our mapping.
    Returns:
        An association table(dictionary) whose key is the word
        and the value the number of occurrences of the word in the corpus passed in parameter.
    """

    # Retrieving the list of words from the corpus
    words = tokenize(corpus)
    # Elimination of duplicates
    unique_words = list(dict.fromkeys(words))

    return {word: occurrence(word, corpus) for word in unique_words}


def tokenize(str) -> list:
    """Returns the list of words from the French corpus passed as an argument.
    Args:
        str: The corpus of which we want the list of words
    Returns:
        A list of words
    """

    # Creation of a tokenizer for the French language.
    french_tokenizer = nltk.RegexpTokenizer(r'''^ \w'|\w+|[^\w\s]|\ w-''')
    # Generation of the token array
    tokens = french_tokenizer.tokenize(str)
    # Filtering on tokens which are words(we ignore punctuation)
    words = [word.lower() for word in tokens if word.isalnum()]
    return words


def occurrence(word, corpus) -> int:
    """Returns the number of occurrences of a word in a corpus.
    Args:
        word: The word for which we want the number of occurrences
        corpus: The corpus in which we want to count occurrences of `word`
    Returns:
        The number of occurrences of the word `word` in the corpus `corpus`
    """
    words = tokenize(corpus)
    return words.count(word)


def extract_keywords(corpus, nb_keywords) -> dict:
    language = "fr"
    max_ngram_size = 4  # we want max 1 words per keyword phrase, no more.

    french_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, top=nb_keywords)
    extracted_keywords = french_kw_extractor.extract_keywords(corpus)

    return {word: score for word, score in extracted_keywords}


def vectorize(sentences):
    vectorizer = TfidfVectorizer()
    return vectorizer, vectorizer.fit_transform(sentences)


def inverse_vectorize(vectorizer, X):
    return vectorizer.inverse_transform(X)


def k_first(k, sentences):
    return sentences[:k]


def k_most_diverse(k, sentences):
    result = []
    if k > 0:
        vectorizer, X = vectorize(sentences)
        inverses = inverse_vectorize(vectorizer, X)
        sentence_vector_dict = {'-'.join(inverses[i].tolist()): sentences[i] for i in range(X.shape[0])}
        kmeans = KMeans(n_clusters=min(k, len(sentences)), random_state=0).fit(X)
        closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
        keys_closest = ['-'.join(closest.tolist()) for closest in inverse_vectorize(vectorizer, X[closest_indices])]
        result = [sentence_vector_dict[k] for k in keys_closest]

    return result


def k_highest_scoring(k, sentences):
    corpus = ';'.join(sentences)
    keywords = extract_keywords(corpus, nb_keywords=word_count(corpus))

    sorted_sentences = sorted(
        sentences,
        # order by node score
        key=lambda sentence: keyword_scoring(sentence, keywords),
        reverse=True
    )

    return sorted_sentences[:k]


def keyword_scoring(sentence, keywords):
    sentence_score = 0
    for keyword, score in keywords.items():
        if keyword in sentence:
            sentence_score += 1/score

    return sentence_score


def google_extract_keywords(corpus, lookup=None):
    if not lookup:
        client = language_v1.LanguageServiceClient()
        document = language_v1.Document(content=corpus, type_=language_v1.Document.Type.PLAIN_TEXT, language='en')
        data = client.analyze_entities(document=document, encoding_type='UTF32')
        counter = Counter([e.name.lower() for e in data.entities])
        keywords = {k: v for k, v in sorted(counter.items(), key=lambda item: item[1], reverse=True) if
                    not k.isnumeric()}
        keywords = list(keywords.keys())

    else:
        keywords = [kw for kw in lookup if kw in corpus]

    return [kw for kw in keywords if len(kw) > 2]
