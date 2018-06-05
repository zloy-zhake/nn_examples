with open("data/noun-lexicon") as f:
    lines_from_lexicon = f.readlines()
with open("data/nouns-from-apertium") as f:
    nouns_from_apertium = f.readlines()

train_data = []

for noun in nouns_from_apertium:
    noun = noun[:noun.find(':')]
    for line in lines_from_lexicon:
        sep_ind = line.find(':')
        word = line[:sep_ind]
        tags = line[sep_ind + 1:].strip()

        if noun in word:
            train_data.append(noun + ':' + tags + ':' + word + '\n')
with open("data/gen-data", 'w') as f:
    f.writelines(train_data)
