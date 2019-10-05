from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

sent = "This is an example showing off stop word filtration."
stop_words = set(stopwords.words("english"))

words = word_tokenize(sent)

filtered_sentence = [w for w in words if not w in stop_words]
print(filtered_sentence)