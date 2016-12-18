#!/bin/bash

ASSIGNMENT="DL-Project-Grønbech-Vording"

mkdir -p $ASSIGNMENT

## Source files
mkdir $ASSIGNMENT/src/
cp src/*.py $ASSIGNMENT/src/.

## Report
cp tex/report.pdf $ASSIGNMENT/$ASSIGNMENT-Report.pdf

## Presentation
cp presentation.pdf $ASSIGNMENT/$ASSIGNMENT-Presentation.pdf

zip -r ../$ASSIGNMENT.zip $ASSIGNMENT

rm -rf $ASSIGNMENT

cp Report/assignment2.pdf ../$ASSIGNMENT.pdf
