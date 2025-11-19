@echo off
REM Quick fix for .env file to use correct Claude model

echo Fixing .env file...

REM Backup existing .env
if exist .env (
    copy .env .env.backup >nul 2>&1
    echo Backed up .env to .env.backup
)

REM Use PowerShell to replace the model names
powershell -Command "(Get-Content .env) -replace 'claude-3-5-sonnet-20241022', 'claude-sonnet-4-5-20250929' | Set-Content .env"
powershell -Command "(Get-Content .env) -replace 'claude-4-5-sonnet-20250929', 'claude-sonnet-4-5-20250929' | Set-Content .env"
powershell -Command "(Get-Content .env) -replace 'LLM_MAX_TOKENS=4096', 'LLM_MAX_TOKENS=16384' | Set-Content .env"

echo.
echo ✅ Fixed! Your .env now uses: claude-sonnet-4-5-20250929
echo.
echo Run: uv run main.py
pause
