"""M7 Train structure search; optional frozen nearest-centroid Validation extension."""
from pathlib import Path
import numpy as np
import pandas as pd
from .configuration import load_config, validate_config
from .datasets import load_train_validation
from .expected_runs import _candidates
from .preprocessing import Standardizer
from .classification import _state
from .clustering import KMeansClustering, AgglomerativeClustering, nearest_centroid
from .metrics import clustering_metrics, train_cluster_mapping, classification_metrics, select_candidate, measure
from .artifacts import write_json, file_sha256
from .persistence import (create_run_directory, save_npz, load_npz, save_reference, load_reference,
    verify_reload, capture_artifacts, transition)
from .m6_training import _numeric_arrays, _record_failure
from .workflow import preflight, provenance, manifest_for, validate_inputs

CLUSTERING_MODELS = ('kmeans','agglomerative')
_CLASSES = {'kmeans':KMeansClustering,'agglomerative':AgglomerativeClustering}


def load_clusterer(path):
    path = Path(path)
    if (path/'model.joblib').exists():
        return load_reference(path/'model.joblib',expected_sha256=file_sha256(path/'model.joblib'))
    arrays,state = load_npz(path/'model.npz')
    expected = _numeric_arrays(state)
    if set(arrays)!=set(expected) or not all(np.array_equal(arrays[k],expected[k]) for k in arrays):
        raise ValueError('Model NPZ/JSON state mismatch')
    return _CLASSES[state['name']].from_dict(state['model'])


def _centers(model,X):
    return np.array([X[model.labels_==k].mean(axis=0) for k in range(len(np.unique(model.labels_)))])


def _fit(name,params,X,config,implementation):
    fixed = config['models'][name]['parameters']
    if implementation=='scratch':
        if name=='kmeans': return KMeansClustering(params['k'],max_iter=fixed['max_iter'],n_init=fixed['n_init'],tol=fixed['tol'],random_state=config['experiment']['seed']).fit(X)
        return AgglomerativeClustering(params['k'],params['linkage']).fit(X)
    from sklearn.cluster import KMeans, AgglomerativeClustering as ReferenceAgglomerative
    if name=='kmeans': return KMeans(n_clusters=params['k'],init=fixed['init'],n_init=fixed['n_init'],max_iter=fixed['max_iter'],tol=fixed['tol'],random_state=config['experiment']['seed']).fit(X)
    return ReferenceAgglomerative(n_clusters=params['k'],linkage=params['linkage'],metric=fixed['metric'],compute_distances=True).fit(X)


def _assign(path,dates,labels,distance,method):
    data = pd.DataFrame({'Date':np.asarray(dates).astype('datetime64[D]').astype(str),'cluster_id':labels,
                         'distance':distance,'assignment_method':method}).to_csv(index=False,lineterminator='\n')
    with path.open('x',encoding='utf8',newline='') as stream: stream.write(data)


def _figure(path,sizes):
    # Exact cluster sizes; fixed zero origin, labelled counts, no learned projection.
    width = 420
    maximum = max(sizes.values())
    rows = []
    for i,(cluster,count) in enumerate(sizes.items()):
        y = 36+i*35
        rows.append(f'<text x="8" y="{y+14}">Cluster {cluster}</text><rect x="100" y="{y}" width="{280*count/maximum}" height="20" fill="#285f99"/><text x="{105+280*count/maximum}" y="{y+14}">{count}</text>')
    path.parent.mkdir()
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{60+len(sizes)*35}" role="img"><title>Train cluster sizes</title><text x="8" y="20">Train cluster sizes (rows, zero origin)</text>'+''.join(rows)+'</svg>',encoding='utf8')


def train_clusterer(name,variant,output_root,run_id,*,config=None,inputs=None,hashes=None,implementation='scratch',extension=False):
    if name not in CLUSTERING_MODELS: raise ValueError('Unknown clustering model')
    preflight(output_root,variant,'clustering',name,[implementation],run_id)
    config = load_config() if config is None else config
    validate_config(config)
    inputs = load_train_validation(variant) if inputs is None else inputs
    computed,runtime = provenance(config,inputs)
    hashes = computed if hashes is None else hashes
    path = create_run_directory(Path(output_root),variant,'clustering',name,implementation,run_id)
    manifest = manifest_for(path,config,inputs,hashes,'clustering',name,implementation,
        lifecycle='clustering_extension' if extension else 'clustering_train')
    transition(manifest,'running',stage='input_validation')
    try:
        validate_inputs(inputs,variant,config)
        write_json(path/'config.json',config)
        write_json(path/'runtime.json',runtime)
        scaler = Standardizer(config['preprocessing']['standardizer']['variance_floor']).fit(inputs.train.X)
        X = scaler.transform(inputs.train.X)
        records,models = [],{}
        manifest['stage'] = 'train_structure_search'
        for i,params in enumerate(_candidates(config['models'][name])):
            record = dict(id=f'c{i:02d}',params=params,metrics=None,error=None)
            try:
                model = _fit(name,params,X,config,implementation)
                record['metrics'] = clustering_metrics(X,model.labels_)
                record['metrics']['stability'] = measure(reason='Optional stability runs deferred until mandatory inventory is complete')
                models[record['id']] = model
            except ValueError as error: record['error'] = str(error)
            records.append(record)
        selected = select_candidate([r for r in records if r['error'] is None],task='clustering')
        model = models[selected['id']]
        centers = _centers(model,X)
        write_json(path/'search_results.json',{'selection_rule':config['selection']['clustering'],'selection_split':'train','selected':selected,'candidates':records})
        write_json(path/'internal_metrics.json',selected['metrics'])
        mapping = train_cluster_mapping(model.labels_,inputs.train.y_classification)
        write_json(path/'cluster_summary.json',{'cluster_sizes':selected['metrics']['cluster_sizes'],
            'centers_standardized':centers.tolist(),'train_only_majority_mapping':{str(k):v for k,v in mapping.items()},
            'target':config['data']['classification_target'],'mapping_tie_class':config['selection']['clustering']['mapping_tie_class']})
        distance = np.linalg.norm(X-centers[model.labels_],axis=1)
        _assign(path/'train_assignments.csv',inputs.train.dates,model.labels_,distance,
                'lloyd_nearest_centroid' if name=='kmeans' else 'train_linkage_tree')
        _figure(path/'figures/cluster_sizes.svg',selected['metrics']['cluster_sizes'])
        manifest['stage'] = 'persisting'
        if implementation=='scratch':
            state = {'name':name,'model':model.to_dict()}
            save_npz(path/'model.npz',_numeric_arrays(state),state)
        else: save_reference(path/'model.joblib',model)
        state = {'kind':'standardize','scaler':_state.scaler_state(scaler),'feature_order':config['data']['feature_columns']}
        save_npz(path/'preprocessor.npz',_numeric_arrays(state),state)
        restored = load_clusterer(path)
        arrays,stored = load_npz(path/'preprocessor.npz')
        expected = _numeric_arrays(stored)
        restored_scaler = _state.scaler_from_state(stored['scaler'])
        receipt = {'labels':verify_reload(model.labels_,restored.labels_,labels=True),
            'centers':verify_reload(centers,_centers(restored,restored_scaler.transform(inputs.train.X))),
            'preprocessor':verify_reload(X,restored_scaler.transform(inputs.train.X)),
            'state_arrays':{'passed':set(arrays)==set(expected) and all(np.array_equal(arrays[k],expected[k]) for k in arrays)}}
        if name=='kmeans': receipt['train_predict'] = verify_reload(model.predict(X),restored.predict(X),labels=True)
        else: receipt['children'] = verify_reload(model.children_,restored.children_,labels=True)
        if extension:
            V = scaler.transform(inputs.validation.X)
            labels,distances = nearest_centroid(V,centers)
            after,_ = nearest_centroid(restored_scaler.transform(inputs.validation.X),_centers(restored,X))
            receipt['extension'] = verify_reload(labels,after,labels=True)
            _assign(path/'validation_assignments.csv',inputs.validation.dates,labels,distances,'frozen_nearest_centroid_extension')
            write_json(path/'extension.json',{'method':'frozen_nearest_centroid','fit_split':'train','evaluation_split':'validation','standard_algorithm_predict':name=='kmeans'})
            write_json(path/'extension_metrics.json',{'post_hoc_train_mapping':True,'metrics':classification_metrics(inputs.validation.y_classification,np.array([mapping[int(k)] for k in labels]))})
            write_json(path/'limitations.json',{'agglomerative':'Frozen nearest-centroid extension is not standard hierarchical predict; no refit on Validation.',
                'test_accessed':False,'cluster_ids':'No semantic identity across variants is assumed.'})
        receipt['passed'] = all(r['passed'] for r in receipt.values())
        write_json(path/'load_verification.json',receipt)
        manifest['load_verification'] = receipt
        manifest['selected_hyperparameters'] = selected['params']
        manifest['output_policy'] = {'selection_split':'train','seed':config['experiment']['seed'],'extension':extension,
            'preprocessing':'Train-only standardization; PCA remains an RF/SVM mechanism'}
        capture_artifacts(manifest,path)
        transition(manifest,'completed',run_dir=path,stage='train_structure_complete')
    except Exception as error:
        _record_failure(path,manifest,error)
        raise
    write_json(path/'manifest.json',manifest)
    return path
