FOR %%A IN (%*) DO ffmpeg -i %%A -ar 8000 -ac 1 %0\..\converted\%%~nA.au 