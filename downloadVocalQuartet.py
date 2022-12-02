import requests, zipfile, io, os, soundfile, shutil
from os.path import join as pjoin

my_tmpdir = 'tmp_download_and_cut_sqam'
os.mkdir(my_tmpdir)
print('Downloading EBU SQAM material ...')
r = requests.get(
    'https://tech.ebu.ch/files/live/sites/tech/files/shared/testmaterial/SQAM_FLAC.zip'
)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall(my_tmpdir)

FS = 44100
start_time = int(21.5 * FS)
end_time = int(23 * FS)
s, fs_sig = soundfile.read(os.path.join(my_tmpdir, '48.flac'))
assert fs_sig == FS
s = s[start_time:end_time, 0]
soundfile.write(pjoin('Utility', 'AudioInputFiles', 'Vocal.wav'), s, FS)

try:
    shutil.rmtree(my_tmpdir)
except OSError as e:
    print("Error: %s - %s." % (e.filename, e.strerror))

print('Downloaded and moved file succesfully to ./Utility/AudioInputFiles/')