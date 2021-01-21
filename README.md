Large-Scale End-to-End Multilingual Speech Recognition and Language Identification with Multi-Task Learning Demo
=====================================================



Overview
--------
This repository contains demo codes for the paper [Large-Scale End-to-End Multilingual Speech Recognition and Language Identification with Multi-Task Learning](http://www.interspeech2020.org/uploadfile/pdf/Mon-3-1-3.pdf). It consists of the trained models, the python inference codes, and a simple frontend webpage as well as a backend nodejs server.


How To Use
----------------------------------------
Use the demo with the frontend webpage
1. clone from https://github.com/Porridge144/sup-mlt-demo.git
2. __cd model_export__ and run python pyonnxrt.py (you might need to run it in background or in a tmux window as it is blocking)
3. __cd server__ and run __node server.js__ (you can change the listening port to an arbitrary one in __server.js__)


Direct inference without using the frontend webpage

1. clone from https://github.com/Porridge144/sup-mlt-demo.git
2. put intended mp3/wav s into __model_export/feat_extract/preprocdir/rawmp3__
3. __cd model_export__ and run python pyonnxrt.py (you might need to run it in background or in a tmux window as it is blocking)
4. __cd model_export/feat_extract__ and run bash run.sh
5. output will be saved in the __server__ and also printed in the terminal which pyonnxrt.py is running


About Author
-------------
- [Shinozaki Lab TITech](http://www.ts.ip.titech.ac.jp/)
