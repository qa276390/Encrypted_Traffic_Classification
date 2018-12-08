source_dir=$1
output_file=$2

find $source_dir -maxdepth 2 -type f -name "*.json" | parallel --verbose "python3 Table_Generator.py --source_data_path {} --output_path $output_file --is_malware $3"

