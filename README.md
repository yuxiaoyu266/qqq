论文名字
Introduce
This is an implementation of the algorithm for creating the -- "Blind Image Quality Assessment with Active Inference" (i.e., AIGQA) method.

Requirements
Python 3.6 or later with all requirements.txt dependencies installed. To install run:

$ pip install -r requirements.txt
Inference
demo.pyruns inference on a variety of sources.To run inference on example images in img/patch:

$ python demo.py --model ../models/best.pkl --patch ../img/patch --dmos ../img/dmos/DMOS.mat
or

$ python demo.py
About Us
Contact
Please contact jupoma@stu.xidian.edu.cn if any issues with the code.
