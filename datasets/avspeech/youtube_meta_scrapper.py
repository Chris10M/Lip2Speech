import os
import pickle
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors

if os.path.isfile('creds.bin'):
    with open('creds.bin', 'rb') as pickle_file:
        YOUTUBE_CREDS = pickle.load(pickle_file)


def meta_scrapper(video_ids):
    api_service_name = "youtube"
    api_version = "v3"

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, credentials=YOUTUBE_CREDS)

    request = youtube.videos().list(
        part='snippet',
        id=",".join(video_ids)
    )
    response = request.execute()

    return response
