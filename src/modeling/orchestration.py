"""Shared M3-M7 router. Train/Validation only; production execution is always explicit."""
import argparse
import copy
import json
from pathlib import Path
import sys
from .configuration import load_config, seal_config
from .datasets import load_train_validation
from .classification_training import M5_MODELS, train_classifier
from .m6_training import M6_MODELS, train_m6_model, candidates_for
from .regression_training import REGRESSION_MODELS, train_regressor
from .clustering_training import CLUSTERING_MODELS, train_clusterer
from .workflow import preflight, run_path, status, provenance, validate_inputs

FAMILIES = {'M3':('multiple_linear',),'M4':REGRESSION_MODELS,'M5':M5_MODELS,'M6':M6_MODELS,'M7':CLUSTERING_MODELS}


def _config(seed):
    config = copy.deepcopy(load_config())
    if seed is not None:
        if not 0<=seed<2**32: raise ValueError('Seed must be an unsigned 32-bit integer')
        config['experiment']['seed'] = seed
        config['neural']['refit_seed'] = seed
        for reference in config['references'].values():
            if isinstance(reference,dict) and 'random_state' in reference.get('parameters',{}):
                reference['parameters']['random_state'] = seed
        config = seal_config(config)
    return config


def _m5_reference(name,variant,root,run_id,config,inputs,original_dates,hashes):
    # M5 behavioral reference follows scratch-selected hyperparameters, as the existing M5 contract specifies.
    # The scratch search is in memory; reference-only requests create no scratch lifecycle/artifacts.
    from .classification_training import fit_candidate, score_of, validation_metrics
    from .classification_reference import run_reference
    from .expected_runs import _candidates
    from .metrics import select_candidate
    candidates,models = [],{}
    for i,params in enumerate(_candidates(config['models'][name])):
        model,protocol = fit_candidate(name,params,inputs.train,original_dates,config)
        score = score_of(name,model,inputs.validation.X)
        record = dict(id=f'c{i:02d}',params=params,protocol=protocol,
            metrics=validation_metrics(name,config,inputs.validation.y_classification,score,0 if name=='perceptron' else .5))
        candidates.append(record)
        models[record['id']] = score
    selected = select_candidate(candidates,task='classification')
    run_reference(name,selected['params'],config,inputs.train,inputs.validation,models[selected['id']],selected['protocol'],
        output_root=root,variant=variant,run_id=run_id,hashes=hashes)
    return run_path(root,variant,'classification',name,'reference',run_id)


def execute(args):
    config = _config(args.seed)
    if args.milestone not in FAMILIES or args.model not in FAMILIES[args.milestone]:
        raise ValueError('Unknown model or model does not belong to milestone')
    task = config['models'][args.model]['task']
    roles = ('scratch','reference') if args.implementation=='both' else (args.implementation,)
    for role in roles: run_path(args.output_root,args.variant,task,args.model,role,args.run_id)
    if args.extension and args.milestone!='M7': raise ValueError('Extension is a clustering lifecycle only')
    if args.command=='dry-run' or args.dry_run:
        result = {'action':'dry-run','milestone':args.milestone,'model':args.model,'variant':args.variant,
            'implementations':roles,'run_id':args.run_id,'seed':config['experiment']['seed'],
            'pilot_only':args.milestone=='M3','test_accessed':False,'config_id':config['config_id']}
        print(json.dumps(result,allow_nan=False))
        return 0
    if args.command=='status':
        result = {role:status(args.output_root,args.variant,task,args.model,role,args.run_id,config=config,loader=load_train_validation) for role in roles}
        print(json.dumps(result,allow_nan=False))
        return 0 if all(r['status'] in ('missing','completed') for r in result.values()) else 1
    preflight(args.output_root,args.variant,task,args.model,roles,args.run_id)
    inputs = load_train_validation(args.variant)
    validate_inputs(inputs,args.variant,config)
    hashes,_ = provenance(config,inputs)
    original_dates = inputs.train.dates
    if args.variant=='non_spike' and args.milestone in ('M5','M6'):
        original_dates = load_train_validation('with_spike').train.dates
    if args.milestone in ('M3','M4'):
        paths = [train_regressor(args.model,args.variant,args.output_root,args.run_id,config=config,inputs=inputs,hashes=hashes,
            implementation=role,pilot_only=args.milestone=='M3') for role in roles]
    elif args.milestone=='M5':
        if args.implementation=='reference':
            paths = [_m5_reference(args.model,args.variant,args.output_root,args.run_id,config,inputs,original_dates,hashes)]
        else:
            paths = [train_classifier(args.model,args.variant,args.output_root,args.run_id,config=config,inputs=inputs,
                original_dates=original_dates,hashes=hashes,reference=args.implementation=='both')]
    elif args.milestone=='M6':
        if args.implementation=='reference':
            from .m6_reference import train_m6_reference
            paths = [train_m6_reference(args.model,args.variant,args.output_root,args.run_id,config=config,inputs=inputs,
                original_dates=original_dates,hashes=hashes,candidates=candidates_for(args.model,config))]
        else:
            paths = [train_m6_model(args.model,args.variant,args.output_root,args.run_id,config=config,inputs=inputs,
                original_dates=original_dates,hashes=hashes,reference=args.implementation=='both')]
    else:
        paths = [train_clusterer(args.model,args.variant,args.output_root,args.run_id,config=config,inputs=inputs,hashes=hashes,
            implementation=role,extension=args.extension) for role in roles]
    print(json.dumps({'paths':[str(p) for p in paths],'test_accessed':False},allow_nan=False))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('train','status','dry-run'))
    parser.add_argument('--milestone',required=True,choices=tuple(FAMILIES))
    parser.add_argument('--model',required=True)
    parser.add_argument('--variant',choices=('with_spike','non_spike'),default='with_spike')
    parser.add_argument('--implementation',choices=('scratch','reference','both'),default='both')
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--output-root','--out-root',type=Path,required=True)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--extension',action='store_true',help='M7 explicit frozen nearest-centroid Validation extension')
    args = parser.parse_args(argv)
    try: return execute(args)
    except (ValueError,OSError,RuntimeError) as error:
        print(f'{type(error).__name__}: {error}',file=sys.stderr)
        return 1


if __name__=='__main__': raise SystemExit(main())
