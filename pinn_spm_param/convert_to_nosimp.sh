# DONT FORGET BayesianCalibration_spm

declare -a fileArrayToLeave=(\
                     )

function list_include_item {
  local list="$1"
  local item="$2"
  if [[ $list =~ (^|[[:space:]])"$item"($|[[:space:]]) ]] ; then
    # yes, list include item
    result=0
  else
    result=1
  fi
  return $result
}

# Files to operate on
fileArray=($(grep -rl "\-simp" . | grep -v "__pycache__" | grep -v "README" | grep -v "argument.py" | grep -v "convert_to_" | grep -v "Model" | grep -v "Log" | grep -v "slurm"))
length=${#fileArray[@]}

# Iterate the string array
for (( i=0; i<${length}; i++ ));
do
    file="${fileArray[$i]}"
    if ! `list_include_item "$fileArrayToLeave" "${fileArray[$i]}"` ; then
      echo $file
      sed -i.backup 's/\-simp/\-nosimp/g' $file
      rm $file.backup
    fi
done


