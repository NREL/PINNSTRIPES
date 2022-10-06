dataFolder=dummy
rm -r Model*
rm -r Log*
poetry update
poetry run python main.py -df $dataFolder -opt -i input
