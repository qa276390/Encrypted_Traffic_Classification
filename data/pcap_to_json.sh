source_dir=$1
joutput_dir=$2
mkdir -p $joutput_dir
soutput_dir=$3
mkdir -p $soutput_dir
#IFS={'\n'}
for i in $(find $source_dir -maxdepth 5 -type f -name "*.pcap")
do
	echo $i
	filename=$(basename "$i" ".pcap")
	echo $filename
	jjsonpath="$joutput_dir/$filename.json"
	sjsonpath="$soutput_dir/$filename.json"
	echo $i
	echo $jjsonpath
	echo $sjsonpath
	
	~/joy/bin/joy tls=1 dns=1 http=1 bidir=1 idp=16 dist=1 entropy=1 num_pkts=$4 "$i" | gunzip > $jjsonpath
	~/joy/sleuth $jjsonpath > $sjsonpath
	
done

for i in $(find $source_dir -maxdepth 5 -type f -name "*.pcapng")
do
	filename=$(basename "$i" ".pcapng")
	
	jjsonpath="$joutput_dir/$filename.json"
	sjsonpath="$soutput_dir/$filename.json"
	echo $i
	echo $jjsonpath
	echo $sjsonpath
	
	~/joy/bin/joy tls=1 dns=1 http=1 bidir=1 idp=16 dist=1 entropy=1 num_pkts=$4 $i | gunzip > $jjsonpath
	~/joy/sleuth $jjsonpath > $sjsonpath
    
done
#unset IFS

