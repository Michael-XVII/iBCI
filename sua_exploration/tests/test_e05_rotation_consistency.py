import numpy as np
from scripts.eval_e05_rotation_consistency import rotated_side, rotate_rows
def test_zero_rotation_and_invariants():
 x=np.array([[1.,2.,3.,4.]],np.float32);m=np.zeros(4,np.float32);s=np.ones(4,np.float32)
 assert np.array_equal(rotated_side(x,m,s,0.,True),x)
 y=rotated_side(x,m,s,.4,True);assert np.array_equal(y[:,2:],x[:,2:])
def test_rotation_preserves_norm():
 x=np.array([[3.,4.]],np.float32);assert np.allclose(np.linalg.norm(rotate_rows(x,.7),axis=1),5.)
