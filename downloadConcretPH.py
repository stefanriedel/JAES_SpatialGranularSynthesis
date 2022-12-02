from __future__ import unicode_literals
import youtube_dl, os
from os.path import join as pjoin

ydl_opts = {
    'format':
    'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['https://www.youtube.com/watch?v=mB0xApHGyWg'])

filename = 'Concret PH-mB0xApHGyWg.mp3'
os.replace(filename, pjoin('Utility', 'AudioInputFiles', filename))

print('Downloaded and moved file succesfully to ./Utility/AudioInputFiles/')
