@echo off
REM TÜBİTAK RAG Projesi Başlatma Scripti (Windows)

echo ========================================
echo TÜBİTAK RAG Sistemi Başlatılıyor...
echo ========================================
echo.

REM API Key kontrolü
if "%GOOGLE_API_KEY%"=="" (
    echo HATA: GOOGLE_API_KEY environment variable tanimlanmamis!
    echo.
    echo Lutfen asagidaki komutu calistirin:
    echo $env:GOOGLE_API_KEY="your_api_key_here"
    echo.
    pause
    exit /b 1
)

echo API Key bulundu: %GOOGLE_API_KEY:~0,10%...
echo.

REM Streamlit uygulamasını başlat
echo Streamlit uygulamasi baslatiliyor...
streamlit run app.py

pause
