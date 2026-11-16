#!/usr/bin/env python3
"""Check bundle and installed protocol with no torch or third-party dependencies."""
import csv, hashlib, json, sys
from pathlib import Path
from collections import Counter
p=Path(sys.argv[1] if len(sys.argv)>1 else Path(__file__).parent/'protocol')
proto=json.loads((p/'protocol.json').read_text())
sha=lambda name:hashlib.sha256((p/name).read_bytes()).hexdigest()
assert sha('labels_all.csv')==proto['labels_all_sha256']
assert sha('dataset_sources.json')==proto['source_index_sha256']
labels={r['slide_id']:r for r in csv.DictReader((p/'labels_all.csv').open())}
assert len(labels)==393
assert Counter(r['label'] for r in labels.values()) == {'0':233,'1':160}
assert Counter(r['source'] for r in labels.values()) == {'official_train':264,'official_test':129}
test=Counter()
for info in proto['folds']:
    f=info['fold']; fname=info['file']
    assert sha(fname)==info['sha256']
    rows=list(csv.DictReader((p/fname).open()))
    sets={k:[r[k] for r in rows if r[k]] for k in ('train','val','test')}
    assert sum(map(len,sets.values()))==len(labels)
    assert set.union(*(set(v) for v in sets.values()))==set(labels)
    assert all(len(v)==len(set(v)) for v in sets.values())
    for a in ('train','val','test'):
        for b in ('train','val','test'):
            if a!=b: assert not set(sets[a])&set(sets[b])
        hist=Counter(labels[s]['label'] for s in sets[a])
        assert hist['0']>0 and hist['1']>0
        assert sum(labels[s]['source']=='official_test' for s in sets[a])==info['stats'][a]['original_test_wsis']
    test.update(sets['test'])
    print(f"fold {f}: train={len(sets['train'])}, val={len(sets['val'])}, test={len(sets['test'])}, classes(test)={Counter(labels[s]['label'] for s in sets['test'])}")
assert test == Counter({sid:1 for sid in labels})
assert proto['seed']==123
print('PASS: all 393 slides appear in test once; no duplicates; class/source balanced; hashes verified')
