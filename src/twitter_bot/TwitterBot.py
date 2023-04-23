import configparser
import tweepy
import pandas as pd
from NewsScraper import NewsScraper
from summarizer import Summarizer
import time


def get_url(tw):
    """Extracts the URL of the Tweet or returns error """
    if len(tw.entities['urls']) == 1:
        return tw.entities['urls'][0]['expanded_url']
    else:
        return 'Not any Url or Several URLS'


# This class serves as an active listener of the notifications that arrive mentioning the bot, if he detects a new it will summarize it
# and afterwards publish its summary on Tw
class TwitterBot:
    def __init__(self):
        # Keys initialitation
        config = configparser.ConfigParser()
        config.read('twitter_bot/keys.ini')
        keys = config['secret']

        auth = tweepy.OAuth1UserHandler(keys['consumer_key_public'], keys['consumer_key_private'],
                                        keys['access_token_public'], keys['access_token_private'])

        # Login in twitter and a queue to track our mentions and the summaries we already did, we track the tweet id
        self.api = tweepy.API(auth)
        self.ns = NewsScraper()
        self.mentions_queue = []  # list of tweet objects to which we have to reply yet
        self.replies_list = pd.read_csv(
            'twitter_bot/replies.csv')  # df containing a column id with the ids of the tw already replied
        self.model = Summarizer()  # Model to summarize the texts
        self.screen_name = self.api.verify_credentials().screen_name

    def thread(self, tweet, text):
        """Given a thread, it tweets making a thread"""
        thread_tweet = self.api.update_status(status=text,
                                              in_reply_to_status_id=tweet.id,
                                              auto_populate_reply_metadata=True)
        return thread_tweet

    def output_summary(self, tweet, text):
        """
        Given a summary it tweets it replying to the user, making a thread if the text is longer than 280 chars
        :param text: text to tweet
        :param tweet: tweet mentioning our bot
        :param media: if pictures are needed, pass a list
        :return: tweet object
        """
        if len(text) <= 270:
            self.thread(tweet, text)
        # Distribute the sentence into readable threads
        sentences = text.split('. ')
        while sentences:
            to_tweet, n = [], 0
            # We form threads of readable sentences
            if n + len(sentences[0]) < 270:
                to_tweet.append(sentences[0])
                n += len(sentences[0])
                sentences.pop(0)
            elif len(sentences[0]) > 270:
                # If we have a too big sentence we just break it into two
                to_break = sentences.pop(0)
                sentences.insert(0, to_break[len(to_break) / 2:len(to_break)])
                sentences.insert(0, to_break[0:len(to_break) / 2])
            else:
                # When we have a big tweet we tweet it
                tweet = self.thread(tweet, '. '.join(to_tweet))

    def check_mentions(self):
        if len(self.replies_list) > 0:
            mentions = self.api.mentions_timeline(max_ids=self.replies_list[-1], count=50)
        else:
            mentions = self.api.mentions_timeline()
        for tw in mentions:
            if tw.id in self.replies_list.id.to_list() or tw in self.mentions_queue \
                    or tw.user.screen_name != self.screen_name:
                continue
            else:
                self.mentions_queue.append(tw)

    def summarize_new(self, url):
        """Performs a summary after scraping URL or returns error"""
        if url == 'Not any Url or Several URLS':
            return url
        try:
            text = self.ns.scrape_new(url)
            summary = self.model(text)
            return summary
        except Exception as e:
            print(e)
            return 'ERROR performing summary'

    def stream(self, time_to_check=60):
        """Daemon that runs summarizing every_tweeet, we can set a time to check the mentions"""
        while True:
            self.check_mentions()
            for mention in self.mentions_queue:
                print(mention)
                url = get_url(mention)
                summary = self.summarize_new(url)
                if summary == url:
                    # if summary = url then we did not receive proper args
                    self.api.update_status(status=url, in_reply_to_status_id=mention.id,
                                           auto_populate_reply_metadata=True)
                elif summary == 'ERROR performing summary':
                    # if summary outputs an error we output other message
                    self.api.update_status(status='I could not make the Summary, I\'m sorry :(',
                                           in_reply_to_status_id=mention.id, auto_populate_reply_metadata=True)
                else:
                    self.output_summary(mention, summary)
                # Update the reply list
                self.replies_list.append(mention.id)
                pd.DataFrame({'id': self.replies_list}).to_csv('replies.csv', index=False)
            print('Mentions finished for now, sleeping')
            time.sleep(time_to_check)


if __name__ == '__main__':
    TwitterBot().stream()
