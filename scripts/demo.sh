dataset=$1
export PYTHONPATH=./
#generating demo from audios from a directory
python main/demo_dir.py --config config/${dataset}/demo.yaml
