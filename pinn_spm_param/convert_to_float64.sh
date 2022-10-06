declare -a fileArray=("util/spm_simpler.py"\
                      "util/_rescale.py"\
                      "util/_losses.py"\
                      "util/custom_activations.py"\
                      "util/pinn.py"\
                      "preProcess/makeDataset_spm.py"\
                      "preProcess/makeDataset_spm_multidata.py"\
                      "postProcess/plotPINNResult.py"\
                      "postProcess/plotData.py"\
                      "postProcess/plotCorrelationPINNvsData.py"\
                      "main.py")

length=${#fileArray[@]}

# Iterate the string array using for loop
for (( i=0; i<${length}; i++ ));
do
    file="${fileArray[$i]}"
    sed -i.bu 's/float32/float64/g' $file
    rm $file.bu
done

