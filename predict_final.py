# This script produces the final submission as shown on
# the kaggle leaderboard.
#
# It runs correctly if placed next to the folders /src
# and /data. The folder /src contains whatever other
# scripts you need (provided by you). The folder /data
# can be assumed to contain two folders /set_train and
# /set_test which again contain the training and test
# samples respectively (provided by user, i.e. us).
#
# Its output is "final_sub.csv"

import os

os.system('mkdir src/preprocessed')
os.system('cd src/ ; python reduce.py')
os.system('cd src/ ; python main.py')
os.system('mv src/final_sub.csv ./final_sub.csv')
