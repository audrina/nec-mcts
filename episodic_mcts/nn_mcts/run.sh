#!/bin/bash
FILE=./20_8Othello
if ! [ -d "$FILE" ]; then
    mkdir $FILE
fi

outputFile="/20_8text.txt"
resultFile="/20_8result.txt"

python3 main.py > $FILE$outputFile
python3 play.py > $FILE$resultFile
python3 plot.py $FILE$resultFile $FILE$outputFile