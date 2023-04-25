call conda create -n saibench python=3.8 -y
call activate saibench
call conda install -c conda-forge cudatoolkit=11.2 cudnn -y
pip install -r ../requirements.txt
pause