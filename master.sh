#!/bin/sh

python3 data_processing/clean_trials.py data/raw/trials/pilot_new.csv data/clean/trials/clean_pilot_trials.csv
# python3 data_processing/summarize_trials.py data/clean/trials/clean_pilot_trials.csv data/clean/trials/summarized_pilot_trials.csv
# python data_processing/summarize_sessions.py data/clean/trials/summarized_pilot_trials.csv data/clean/trials/summarized_pilot_sessions.csv

python3 data_processing/clean_answers.py data/raw/answers/pilot2.csv data/clean/answers/clean_answers.csv