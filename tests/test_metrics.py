import numpy as np
from drd.metrics import evaluate_predictions

def test_metrics_shape():
    y=np.array([0,1,2,3,4,0,1,2,3,4])
    p=np.eye(5)[y]
    result=evaluate_predictions(y,p)
    assert result["accuracy"]==1.0
    assert result["macro_f1"]==1.0
    assert len(result["confusion_matrix"])==5
