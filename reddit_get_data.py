import praw
import pandas as pd

reddit = praw.Reddit(client_id='my_client_id',
                     client_secret='my_client_secret', user_agent='my_user_agent')

posts = []
aita_subreddit = reddit.subreddit('AmItheAsshole')
num = 0

for post in aita_subreddit.controversial(limit=10000):

    submission = reddit.submission(id = post.id)
    submission.comments.replace_more(limit=0)
    comments = []
    
    for comment in submission.comments:
        if "*I am a bot" not in comment.body:
            comments.append(comment.body)


    posts.append([post.title, post.score, post.id, post.subreddit,
                  post.url, len(comments), comments, post.selftext, post.created])
    
    num += 1
    print(num)


posts = pd.DataFrame(posts, columns=[
                     'title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'comments', 'body', 'created'])
posts.to_csv('AmItheAsshole_with_comments.csv')
