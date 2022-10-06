dataFolder=../integration_spm/

echo Generating analytical solution ...
cd $dataFolder
python main.py -p 0.5 1
python main.py -p 0.5 10
python main.py -p 4 1
python main.py -p 4 10
python main.py -p 2 2
cd ../preProcess
echo Done!


echo Generating dataset ...
python makeDataset_spm.py -df $dataFolder -frt 1 -p 0.5 1
python makeDataset_spm.py -df $dataFolder -frt 1 -p 0.5 10
python makeDataset_spm.py -df $dataFolder -frt 1 -p 4 1
python makeDataset_spm.py -df $dataFolder -frt 1 -p 4 10
python makeDataset_spm.py -df $dataFolder -frt 1 -p 2 2
echo Done!


echo Generating multi dataset ...
python makeDataset_spm_multidata.py -df $dataFolder -dl solution_0.5_1.npz solution_0.5_10.npz  solution_4_1.npz solution_4_10.npz -p 0.5 1 0.5 10 4 1 4 10
#python makeDataset_spm_multidata.py -df $dataFolder -dl solution_0.5_1.npz -p 0.5 1
echo Done!

