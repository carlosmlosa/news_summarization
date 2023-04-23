from requests_html import HTMLSession
from bs4 import BeautifulSoup
import nltk
import textstat
import re


class NewsScraper:

    def __init__(self):
        self.session = HTMLSession()

    def get_new(self, url):
        """Gets HTML text for a new"""
        # make a request to the page
        new = self.session.get(url)
        # render the JavaScript
        new.html.render()
        return new

    def get_plain_text(self, new):
        """Removes all HTML labels"""
        soup = BeautifulSoup(new.text, 'lxml')
        text = soup.get_text()
        return text

    def get_readable_text(self, text, threshold=0):
        """Using tokenizing with nltk, it extracts the readable information of a new,
           given a threshold of the flesch readability score"""
        legal_terms = ['Distribution and use of this material', 'All Rights Reserved',
                       'For non-personal use or to order multiple copies']
        words = nltk.word_tokenize(text)
        text = ' '.join(words)
        # Eliminate URLS
        text = re.sub(r'http\S+', '', text)
        # Extract readable sentences
        splits = text.split('. ')
        outputs = []
        for item in splits:
            for term in legal_terms:
                if term in item:
                    break
            else:
                if textstat.flesch_reading_ease(item) > threshold:
                    outputs.append(item)
        return '. '.join(outputs)

    def scrape_new(self, url):
        """Given an url, it extracts the new"""
        new = self.get_new(url)
        if new.status_code != 200:
            return 'I could not extract the information'
        else:
            text = self.get_plain_text(new)
            return self.get_readable_text(text)
