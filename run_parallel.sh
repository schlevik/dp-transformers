

path=/home/srini/dp-transformers/

$path/run_generate_v2.sh 0.5 3 &
$path/run_generate_v2.sh 1 5 &
$path/run_generate_v2.sh 2 6 &
$path/run_generate_v2.sh 4 7 &

wait
echo "All jobs completed"









