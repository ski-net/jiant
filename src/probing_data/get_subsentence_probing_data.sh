wget http://decomp.net/wp-content/uploads/2015/08/protoroles_eng_pb.tar.gz
tar zxvf protorole_eng_pb.tar.gz

wget http://decomp.net/wp-content/uploads/2015/08/protoroles_eng_ud1.2.tar.gz
tar zxvf protoroles_eng_ud1.2.tar.gz

wget https://github.com/UniversalDependencies/UD_English/archive/r1.2.tar.gz
tar -zxvf r1.2.tar.gz

mkdir recast1
cd recast1
wget http://decomp.net/wp-content/uploads/2017/11/inference_is_everything.zip
unzip inference_is_everything

#NER
wget http://gmb.let.rug.nl/releases/gmb-2.2.0.zip
unzip gmb-2.2.0.zip

git clone https://github.com/synalp/NER.git
mkdir CoNLL-2003
cp NER/corpus/CoNLL-2003/* CoNLL-2003
rm -rf NER/

