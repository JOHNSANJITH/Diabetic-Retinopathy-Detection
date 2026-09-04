import numpy as np
from PIL import Image
from drd.preprocessing import load_image

def test_load_image_contract(tmp_path):
    path=tmp_path/"sample.jpg"; Image.new("RGB",(80,60),"white").save(path)
    arr=load_image(path)
    assert arr.shape==(299,299,3)
    assert arr.dtype==np.float32
