#! /bin/bash

mkdir data

curl http://spamassassin.apache.org/old/publiccorpus/20021010_easy_ham.tar.bz2 > data/nonspam.tar.bz2
bzip2 -d data/nonspam.tar.bz2
tar -C data/ -zxf data/nonspam.tar
rm data/nonspam.tar
mv data/easy_ham data/nonspam

curl http://spamassassin.apache.org/old/publiccorpus/20021010_spam.tar.bz2 > data/spam.tar.bz2
bzip2 -d data/spam.tar.bz2
tar -C data/ -zxf data/spam.tar
rm data/spam.tar
