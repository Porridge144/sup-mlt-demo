from os import walk

path = "preprocdir/rawmp3"

_, _, filenames = next(walk(path), (None, None, []))

f = open("preprocdir/wav.scp", "w")

for filename in filenames:
    line = filename + " " + "ffmpeg -i preprocdir/rawmp3/" + filename + " -f wav -ar 16000 -ab 16 - |\n"
    f.write(line)