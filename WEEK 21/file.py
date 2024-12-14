import csv

# Data provided as text
data = """
Question,Answer
What is Natural Language Processing (NLP)?,"NLP is a field of artificial intelligence that focuses on the interaction between computers and human language."
What is tokenization?,"Tokenization is the process of breaking down text into smaller units called tokens."
What is stemming?,"Stemming is the process of reducing words to their root form."
What is lemmatization?,"Lemmatization is the process of reducing words to their dictionary form."
What is part-of-speech tagging?,"Part-of-speech tagging is the process of assigning a part of speech to each word in a sentence."
What is named entity recognition (NER)?,"NER is the process of identifying named entities in text, such as people, organizations, and locations."
What is sentiment analysis?,"Sentiment analysis is the process of determining the sentiment expressed in a piece of text, such as positive, negative, or neutral."
What is text classification?,"Text classification is the process of assigning predefined categories to text documents."
What is text generation?,"Text generation is the process of generating text, such as articles, poems, or code."
What is machine translation?,"Machine translation is the process of translating text from one language to another."
What is question answering?,"Question answering is the process of answering questions posed in natural language."
What is text summarization?,"Text summarization is the process of condensing a text document into a shorter version."
What are some popular NLP libraries?,"Some popular NLP libraries include NLTK, spaCy, and Transformers."
What is a language model?,"A language model is a statistical model that predicts the likelihood of a sequence of words."
What is a word embedding?,"A word embedding is a numerical representation of a word."
What is a recurrent neural network (RNN)?,"An RNN is a type of neural network that is well-suited for processing sequential data, such as text."
What is a long short-term memory (LSTM) network?,"An LSTM network is a type of RNN that can learn long-term dependencies in data."
What is a transformer network?,"A transformer network is a type of neural network that is well-suited for processing sequential data, such as text, without relying on recurrent connections."
What is attention mechanism?,"An attention mechanism allows a model to focus on the most relevant parts of the input sequence."
What is the difference between rule-based and statistical NLP?,"Rule-based NLP relies on handcrafted rules, while statistical NLP relies on statistical models."
What is the role of NLP in chatbots?,"NLP enables chatbots to understand and respond to natural language queries."
What is the role of NLP in information retrieval?,"NLP helps improve the accuracy and relevance of search engine results."
What is the role of NLP in text summarization?,"NLP helps identify the most important information in a text document and condense it into a shorter summary."
What are some challenges in NLP?,"Some challenges in NLP include ambiguity, polysemy, and lack of annotated data."
What is the future of NLP?,"The future of NLP holds great promise, with advancements in deep learning and AI leading to more sophisticated and intelligent language processing systems."
"""

# Convert the text data into CSV format
with open("nlp_questions_answers.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    
    # Split the text into lines and write each line
    for line in data.strip().split("\n"):
        writer.writerow(line.split(",", 1))

print("CSV file 'nlp_questions_answers.csv' created successfully!")
