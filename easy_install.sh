cd dep/bc_exploration
git checkout master
git pull --recurse-submodules
cd ../bc_gym_planning_env
git checkout feature/add-initial-raytracing-code
git pull --recurse-submodules
cd ../..
pipenv install
pipenv shell
python setup.py install
