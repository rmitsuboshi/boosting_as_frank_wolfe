path='/PATH/TO/BENCHMARK/DATASET'

files=(
    'banana'
    'breast_cancer'
    'diabetes'
    'german'
    'heart'
    'ringnorm'
    'flare_solar'
    'thyroid'
    'titanic'
    'waveform'
)


ratios=(0.1 0.2 0.3 0.4 0.5)


boosters=(
    'lpb'
    'mlpb(pfw)'
    'mlpb(pfw_only)'
    'mlpb(ss)'
    'mlpb(ss_only)'
    'erlpb'
)


for booster in ${boosters[@]} ; do
    echo $booster
    for file in ${files[@]} ; do
        echo "  ${file}"
        input="${path}/${file}.csv"
        for ratio in ${ratios[@]} ; do
            output="time_result/${booster}_${file}_${ratio}.txt"
            /usr/bin/time -v --output=$output ./time $input $booster $ratio
        done
    done
done




