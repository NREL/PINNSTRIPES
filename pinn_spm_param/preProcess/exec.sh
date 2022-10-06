dataFolder=../integration_spm/

if [ ! -f $dataFolder/solution.npz ]; then
    echo Generating analytical solution ...
    cd $dataFolder
    python main.py    
    cd ../preProcess
    echo Done!
fi

echo Generating dataset ...
python makeDataset_spm.py -df $dataFolder -frt 4
echo Done!
