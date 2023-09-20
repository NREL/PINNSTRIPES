dataFolder=../integration_spm/

if [ ! -f $dataFolder/solution.npz ]; then
    echo Generating analytical solution ...
    cd $dataFolder
    python main.py -nosimp -opt
    cd ../preProcess
    echo Done!
fi

echo Generating dataset ...
python makeDataset_spm.py -nosimp -df $dataFolder -frt 1
echo Done!

