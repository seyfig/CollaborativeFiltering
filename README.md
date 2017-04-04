# Collaborative Filtering

## Overview
Collaborative Filtering is a Python 3.5 application, mainly to recommend movies to the users. The problem and the dataset is taken from [NetFlix Prize](http://www.netflixprize.com/rules). The number of samples in the training dataset is 3.25 million, and the number of samples in the test dataset is 100,000.

## The Project
The user similarity matrix was created in the first step. This step took 9 hours to complete,and it required 9 GB RAM. After the similarity matrix was created, different K values tried to predict ratings. The mean average error (MAE), the root mean square error (RMSE) and the samples for given rounded error are given in the following table.

| K   | MAE      | RMSE     |    0  |    1   |   2  |  3  |  4 | 5 |
|:---:|:--------:|:--------:|:-----:|:------:|:----:|:---:|:--:|:-:|
|  10 | 0.714424 | 0.910386 | 43517 |  47041 | 8913 | 966 | 41 | 0 |
|  20 | 0.697659 | 0.889698 | 44645 |  46654 | 8313 | 835 | 30 | 1 |
|  28 | 0.691211 | 0.881747 | 45018 |  46588 | 8032 | 813 | 27 | 0 |
|  30 | 0.690428 | 0.880855 | 44995 |  46624 | 8021 | 808 | 29 | 1 |
|  40 | 0.687346 | 0.877227 | 45236 |  46513 | 7907 | 788 | 33 | 1 |
|  50 | 0.685667 | 0.875187 | 45255 |  46567 | 7842 | 783 | 30 | 1 |
|  60 | 0.68437  | 0.873731 | 45337 |  46528 | 7809 | 772 | 31 | 1 |
|  70 | 0.683625 | 0.87289  | 45378 |  46545 | 7760 | 765 | 29 | 1 |
|  80 | 0.682981 | 0.872294 | 45570 |  46322 | 7787 | 770 | 28 | 1 |
|  90 | 0.682473 | 0.871747 | 45659 |  46264 | 7768 | 759 | 28 | 0 |
|  91 | 0.682394 | 0.871689 | 45687 |  46226 | 7779 | 759 | 27 | 0 |
| 100 | 0.682154 | 0.871501 | 45668 |  46235 | 7783 | 762 | 30 | 0 |

## Dependencies
* numpy library is required to perform matrix operations.