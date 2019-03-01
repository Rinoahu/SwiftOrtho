#!/bin/bash
#cd SwiftOrtho
#cd FastClust
rm -f ./lib/fsearch-c
rm -rf ./tmpdir ./bin/__pycache__
find ./ -name *un~ | xargs rm
cd example
rm -f test.fsa.sc*
rm -rf test.fsa_results/
rm -rf select.fsa_results/
cd ..

git config --global user.email xiaohu@iastate.edu
git config --global user.name Rinoahu


git remote rm origin

git add -A .
git commit -m 'fixed a bug in binary search'
git remote add origin https://github.com/Rinoahu/SwiftOrtho

git pull origin master
git push origin master

git checkout master
