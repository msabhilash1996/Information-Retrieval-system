# Information Retrieval System

Implement a linear search in the document collection using a single search term.

## Features

- Reads the source file and stop-words file from the working directory
- Performs creation of separate text file for each fables in 2 separate folders ( collection_originals , collection_no_stopwords )

## Tech
Script uses following tech stack:

- [Python] - An interpreted high-level general-purpose programming language.

## Installation

IR system requires [Python](https://www.python.org) 3+ to run.

- Script used : ir_system.py
- Place the scripts into a directory of choice.

## Pre-Requisites

aesopa10.txt, englishST.txt, porter.txt, ground_truth.txt 
The above mentioned file should be present where the python script is running.

## Usage

- Change the directory where the script is placed.
- In the command line for the 1st run execute : python ir_system.py --extract-collection aesopa10.txt
- This will separate the fables into 2 folders collection_originals , collection_no_stopwords 
- To search for the word in the original files, execute: python ir_system.py --query the --documents original
  or python ir_system.py --model bool --search-mode linear --documents original --query <word>
- To search for the word in the no stop-words file, execute: python ir_system.py --query the --documents no_stopwords
  or python ir_system.py --model bool --search-mode linear --documents no_stopwords --query <word>
- To stem the word use the command 
  eg: python ir_system.py --model "bool" --search-mode "linear" --documents "no_stopwords" --query "future" --stemming
- To use vector space model along with boolean seach mode
  eg:python ir_system.py --model "vector" --search-mode "inverted" --documents "no_stopwords" --query "animal|hunters" --stemming
  orpython ir_system.py --model "vector" --documents "no_stopwords" --query "animal|hunters" --stemming
  
Note: Order of arguments doesn't matter




