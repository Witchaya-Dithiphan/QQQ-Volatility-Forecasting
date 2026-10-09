"""M3/M4 frozen regression searches on Train/Validation; scratch core and sklearn reference."""
from pathlib import Path
import numpy as np
from .configuration import load_config, validate_config
from .datasets import load_train_validation
from .expected_runs import _candidates
from .metrics import regression_outputs, select_candidate
from .artifacts import write_json, write_predictions, file_sha256
from .persistence import (create_run_directory, save_npz, load_npz, save_reference, load_reference,
    verify_reload, capture_artifacts, transition)
from .preprocessing import Standardizer
from .classification import _state
from .m6_training import _numeric_arrays, _record_failure
from .workflow import preflight, provenance, manifest_for, validate_inputs
from .regression.simple_linear import SimpleLinearRegression
from .regression.multiple_linear import MultiLinearRegression
from .regression.polynomial import PolynomialRegression
from .regression.elastic_net import ElasticNet

REGRESSION_MODELS = ('simple_linear','multiple_linear','polynomial','elastic_net')
_CLASSES = dict(zip(REGRESSION_MODELS,(SimpleLinearRegression,MultiLinearRegression,PolynomialRegression,ElasticNet)))


class RegressionModel:
    """Owns feature selection and Train-fitted preprocessing exactly once."""
    def __init__(self,name,params,estimator,scaler=None,feature_order=None):
        self.name,self.params,self.estimator,self.scaler = name,dict(params),estimator,scaler
        self.feature_order = list(feature_order)

    def transform(self,X):
        X = np.asarray(X,dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.feature_order): raise ValueError('Feature order/count mismatch')
        if 'feature' in self.params: X = X[:,[self.feature_order.index(self.params['feature'])]]
        return self.scaler.transform(X) if self.scaler else X

    def predict(self,X): return self.estimator.predict(self.transform(X))

    def to_dict(self):
        return {'name':self.name,'params':self.params,'feature_order':self.feature_order,
                'model':self.estimator.to_dict(),'scaler':_state.scaler_state(self.scaler) if self.scaler else None}

    @classmethod
    def from_dict(cls,state):
        return cls(state['name'],state['params'],_CLASSES[state['name']].from_dict(state['model']),
                   _state.scaler_from_state(state['scaler']) if state['scaler'] else None,state['feature_order'])


def fit_candidate(name,params,train,config,implementation):
    fixed = config['models'][name]['parameters']
    scaler = None
    # PolynomialRegression owns standardization before univariate powers.
    if name in ('multiple_linear','elastic_net'):
        scaler = Standardizer(config['preprocessing']['standardizer']['variance_floor']).fit(train.X)
    if implementation == 'scratch':
        kwargs = {'degree':params['degree']} if name == 'polynomial' else (
            {**params,'max_iter':fixed['max_iter'],'tol':fixed['tol']} if name == 'elastic_net' else {})
        estimator = _CLASSES[name](**kwargs)
    else:
        from sklearn.linear_model import LinearRegression, ElasticNet as ReferenceElasticNet
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import make_pipeline
        if name == 'polynomial':
            # Reference standardization is fit on Train, before powers, never on Validation.
            column = config['data']['feature_columns'].index(params['feature'])
            scaler = Standardizer(config['preprocessing']['standardizer']['variance_floor']).fit(train.X[:,[column]])
            estimator = make_pipeline(PolynomialFeatures(params['degree'],include_bias=False),LinearRegression(fit_intercept=True))
        elif name == 'elastic_net': estimator = ReferenceElasticNet(**params,**config['references']['elastic_net']['parameters'])
        else: estimator = LinearRegression(fit_intercept=True)
    model = RegressionModel(name,params,estimator,scaler,config['data']['feature_columns'])
    estimator.fit(model.transform(train.X),train.y_regression)
    return model


def metrics(y,raw,config):
    result = regression_outputs(y,raw,clip_min=config['selection']['regression_clip_min'])
    return {k:v for k,v in result.items() if k not in ('raw_prediction','final_prediction','clipping_indicator')}


def load_regressor(path):
    path = Path(path)
    if (path/'model.joblib').exists():
        return load_reference(path/'model.joblib',expected_sha256=file_sha256(path/'model.joblib'))
    arrays,state = load_npz(path/'model.npz')
    expected = _numeric_arrays(state)
    if set(arrays) != set(expected) or not all(np.array_equal(arrays[k],expected[k]) for k in arrays):
        raise ValueError('Model NPZ/JSON state mismatch')
    return RegressionModel.from_dict(state)


def train_regressor(name,variant,output_root,run_id,*,config=None,inputs=None,hashes=None,implementation='scratch',pilot_only=False):
    if name not in REGRESSION_MODELS: raise ValueError('Unknown regression model')
    preflight(output_root,variant,'regression',name,[implementation],run_id)
    config = load_config() if config is None else config
    validate_config(config)
    if pilot_only and name != config['experiment']['pilot']['model']: raise ValueError('Invalid pilot model')
    inputs = load_train_validation(variant) if inputs is None else inputs
    computed,runtime = provenance(config,inputs)
    hashes = computed if hashes is None else hashes
    path = create_run_directory(Path(output_root),variant,'regression',name,implementation,run_id)
    manifest = manifest_for(path,config,inputs,hashes,'regression',name,implementation,pilot_only=pilot_only)
    transition(manifest,'running',stage='input_validation')
    try:
        validate_inputs(inputs,variant,config)
        write_json(path/'config.json',config)
        write_json(path/'runtime.json',runtime)
        manifest['stage'] = 'candidate_search'
        records,models = [],{}
        for index,params in enumerate(_candidates(config['models'][name])):
            record = dict(id=f'c{index:02d}',params=params,metrics=None,error=None)
            try:
                model = fit_candidate(name,params,inputs.train,config,implementation)
                record['metrics'] = metrics(inputs.validation.y_regression,model.predict(inputs.validation.X),config)
                models[record['id']] = model
            except ValueError as error: record['error'] = str(error)
            records.append(record)
        selected = select_candidate([r for r in records if r['error'] is None],task='regression')
        model = models[selected['id']]
        raw = model.predict(inputs.validation.X)
        write_json(path/'search_results.json',{'selection_rule':config['selection']['regression'],'selected':selected,'candidates':records})
        manifest['stage'] = 'persisting'
        if implementation == 'scratch':
            state = model.to_dict()
            save_npz(path/'model.npz',_numeric_arrays(state),state)
        else: save_reference(path/'model.joblib',model)
        preprocessing = {'kind':'standardize' if model.scaler else ('internal_polynomial' if name=='polynomial' else 'none'),
            'feature_order':model.feature_order,'selected_feature':model.params.get('feature'),
            'scaler':_state.scaler_state(model.scaler) if model.scaler else None}
        if name == 'polynomial' and implementation == 'scratch':
            preprocessing['internal'] = {k:v for k,v in model.estimator.to_dict().items() if k!='coefficients'}
        save_npz(path/'preprocessor.npz',_numeric_arrays(preprocessing),preprocessing)
        restored = load_regressor(path)
        after = restored.predict(inputs.validation.X)
        a,s = load_npz(path/'preprocessor.npz')
        expected = _numeric_arrays(preprocessing)
        receipt = {'raw':verify_reload(raw,after,rtol=config['persistence']['reload_rtol'],atol=config['persistence']['reload_atol']),
            'clipped':verify_reload(np.maximum(raw,0),np.maximum(after,0)),
            'preprocessor':{'passed':s==preprocessing and set(a)==set(expected) and all(np.array_equal(a[k],expected[k]) for k in a)}}
        receipt['passed'] = all(r['passed'] for r in receipt.values())
        write_json(path/'load_verification.json',receipt)
        manifest['load_verification'] = receipt
        manifest['selected_hyperparameters'] = selected['params']
        manifest['output_policy'] = {'clipping_min':config['selection']['regression_clip_min'],'retain_raw':True,
            'seed':config['experiment']['seed'],'target':config['data']['regression_target']}
        result = metrics(inputs.validation.y_regression,raw,config)
        result['baselines'] = {'train_mean':metrics(inputs.validation.y_regression,np.full(len(raw),inputs.train.y_regression.mean()),config),
            'historical_volatility_5d':metrics(inputs.validation.y_regression,
                inputs.validation.X[:,model.feature_order.index('historical_volatility_5d')],config)}
        write_json(path/'validation_metrics.json',result)
        write_predictions(path/'validation_predictions.csv',dates=inputs.validation.dates,target=inputs.validation.y_regression,raw=raw,task='regression')
        capture_artifacts(manifest,path)
        transition(manifest,'completed',run_dir=path,stage='validation_complete')
    except Exception as error:
        _record_failure(path,manifest,error)
        raise
    write_json(path/'manifest.json',manifest)
    return path
