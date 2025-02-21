# app/utils/fetch_sentiment_data.py

import praw
import pandas as pd
from datetime import datetime
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def reddit_authentication():
    reddit = praw.Reddit(client_id=os.getenv('REDDIT_CLIENT_ID'),
                         client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                         user_agent=os.getenv('REDDIT_USER_AGENT'))
    return reddit

def fetch_reddit_posts(reddit, subreddit_name, query, limit=1000):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    print(f"Fetching {limit} posts for query '{query}' from r/{subreddit_name}...")
    try:
        for post in subreddit.search(query, limit=limit, sort='new'):
            posts.append({
                'id': post.id,
                'title': post.title,
                'selftext': post.selftext,
                'created_utc': datetime.utcfromtimestamp(post.created_utc),
                'score': post.score,
                'url': post.url,
                'num_comments': post.num_comments,
                'ticker': query
            })
    except Exception as e:
        print(f"An error occurred while fetching posts for {query}: {e}")
    return pd.DataFrame(posts)

def save_sentiment_data(df, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"Sentiment data saved to '{file_path}'")