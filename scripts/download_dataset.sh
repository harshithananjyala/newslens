#!/usr/bin/env bash
# Downloads the AG News training set (120,000 news articles, about 29 MB) into data/raw/.
# Source: https://github.com/mhjabreel/CharCnn_Keras (a public copy of the AG News corpus)
set -e
cd "$(dirname "$0")/.."
mkdir -p data/raw
curl -L --fail --progress-bar -o data/raw/ag_news_train.csv \
  https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv
echo "Saved data/raw/ag_news_train.csv ($(wc -l < data/raw/ag_news_train.csv | tr -d ' ') articles)"
