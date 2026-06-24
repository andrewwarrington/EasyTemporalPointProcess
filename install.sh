conda create --name=easytpp python=3.12.4
chmod 777 activate.sh
source activate.sh
pip install -r requirements.txt
pip install -e .
