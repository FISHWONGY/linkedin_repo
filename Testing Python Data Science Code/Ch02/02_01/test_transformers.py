import numpy as np
import sys
sys.path.extend(['/Users/yuawong/Documents/GitHub/linkedin_repo/Testing Python Data Science Code/Ch02/02_01'])
import transformers as tr


def test_scale():
    v = np.array([0.1, 1.0, 1.1])
    out = tr.scale(v, 1, 1.1)
    expected = np.array([0.1, 1.1, 1.21])
    assert np.allclose(expected, out)
