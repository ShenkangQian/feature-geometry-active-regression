"""Executable API demonstration using explicitly illustrative observations."""
import argparse
import json
import numpy as np
from geometa.api import ActiveRegressor


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args=parser.parse_args()
    regressor=ActiveRegressor(args.checkpoint,seed=0)
    if regressor.model.output_dim != 1: raise ValueError("Use a pulse checkpoint for this demonstration.")
    regressor.observe(np.array([-1.,0.,1.]),np.array([.1,.8,-.1]))
    prediction=regressor.predict(np.linspace(-1,1,51))
    proposal=regressor.suggest(np.linspace(-1,1,25),np.linspace(-.97,.97,20))
    print(json.dumps({"observation_source":"illustrative, not new experimental evidence",
          "mean_shape":list(prediction["mean"].shape),"covariance_shape":list(prediction["covariance"].shape),
          "suggestion":proposal},indent=2))
