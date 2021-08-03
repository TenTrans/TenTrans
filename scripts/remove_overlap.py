import sys

f1 = open(sys.argv[1], 'r', encoding='utf8').readlines()
f2 = open(sys.argv[2], 'r', encoding='utf8').readlines()

for line1 in f1:
    if line1 not in f2:
        print(line1, end="")
