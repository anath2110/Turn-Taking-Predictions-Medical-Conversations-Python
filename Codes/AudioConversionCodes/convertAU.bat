FOR %%A IN (%*) DO sox %%A -b 16 -e signed-integer %0\..\converted\%%~nA.au channels 1 rate 8k