#!/bin/bash
        for i in `seq 100 234`;
        do
                                curl -u wyklad:parallel 'https://www.physionet.org/physiobank/database/mitdb/'$i'.atr' --output ${i}'.atr'
                                curl -u wyklad:parallel 'https://www.physionet.org/physiobank/database/mitdb/'$i'.dat' --output ${i}'.dat'
                                curl -u wyklad:parallel 'https://www.physionet.org/physiobank/database/mitdb/'$i'.hea' --output ${i}'.hea'
        done
