#!/bin/sh

python data_processing/clean_trials.py data/raw/trials/pilot.csv data/clean/trials/clean_pilot_trials.csv
python data_processing/summarize_trials.py data/clean/trials/clean_pilot_trials.csv data/clean/trials/summarized_pilot_trials.csv