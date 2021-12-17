#!/bin/sh

python data_processing/clean_trials.py data/raw/trials/pilot.csv data/clean/trials/clean_pilot_trials.csv
python data_processing/summarize_trials.py data/clean/trials/clean_pilot_trials.csv data/clean/trials/summarized_pilot_trials.csv
# python data_processing/summarize_sessions.py data/clean/trials/summarized_pilot_trials.csv data/clean/trials/summarized_pilot_sessions.csv

python data_processing/clean_answers.py data/raw/answers/pilot.csv data/clean/answers/clean_answers.csv