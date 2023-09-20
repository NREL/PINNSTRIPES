format () {
    local checkOnly="$2";

    if [ "$checkOnly" = "true" ];
    then

        black --check  --line-length 79 --target-version 'py310' --include  '\.pyi?$' $1
        isort --check-only --diff --profile 'black' --multi-line 3 --trailing-comma --force-grid-wrap 0 --line-length 79 --use-parentheses $1
         
    else

        black --line-length 79 --target-version 'py310' --include  '\.pyi?$' $1
        isort --profile 'black' --multi-line 3 --trailing-comma --force-grid-wrap 0 --line-length 79 --use-parentheses $1
    fi; 
}



