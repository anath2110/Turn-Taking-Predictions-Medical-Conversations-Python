@echo off
@echo ./> %1
@echo.>> %1
FOR %%A IN (%*) DO (
IF  "%%A" NEQ "%~1" @echo l %%A>> %1
)