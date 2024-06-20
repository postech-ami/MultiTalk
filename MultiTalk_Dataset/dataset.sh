for language in "$@"
do
python download_and_process.py --language "$language"
done