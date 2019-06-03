#!/bin/bash

# install python packages
#python -mpip install -U pip numpy==1.16.4 scipy==1.3.0 networkx==2.3 cffi==1.12.3 biopython==1.73 
conda install -y networkx=2.3 cffi=1.12.3 biopython=1.73 numpy=1.16.4 scipy=1.3.0

rm -rf ./pypy
# 1. download portable pypy
wget -c https://bitbucket.org/squeaky/portable-pypy/downloads/pypy-7.1.0-linux_x86_64-portable.tar.bz2

# 2. unzip the compressed file
tar xvf pypy-7.1.0-linux_x86_64-portable.tar.bz2

# 3. remove the compressed file
rm pypy-7.1.0-linux_x86_64-portable.tar.bz2

# 4. install pypy

mv ./pypy-7.1.0-linux_x86_64-portable/ ./pypy
cd pypy
mkdir install_dir
./bin/virtualenv-pypy -p ./bin/pypy ./install_dir

./install_dir/bin/pypy -mpip install -U pip rpython

#pypy/bin/pypy ./bin/find_hit.py &> /dev/null
#./install_dir/bin/pypy ../bin/find_hit.py &> /dev/null
./install_dir/bin/pypy ../bin/find_hit.py






