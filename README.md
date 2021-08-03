# LCLS Cu Injector Model for ML Testing
Working repo for the LCLS Cu line injector NN surrogate model for ML testing

## Environment
```bash
$ conda env create -f injmodel.yml
```
or
```bash
$ conda conda create --name injmodel --file injmodel.txt
```

## Basic Usage
A [simple BO example](https://github.com/slaclab/lcls_cu_injector_ml_model/blob/main/injector_surrogate/injector_emit_prediction_BO_example.ipynb) shows how to interact with the model to optimize on the emittance scalar outputs of the surrogate.

For more details on the injector tuning and the capabilties of the model, see the [lcls_cu_injector_tuning repo](https://github.com/slaclab/lcls_cu_injector_tuning/).
