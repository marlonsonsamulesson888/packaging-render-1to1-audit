import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parents[1]
def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m
new=load(BASE/'scripts/audit_render.py','new_audit')

def fixture(root, case):
    shape=(100,120)
    win=np.zeros(shape,np.uint8);win[:50,:]=255
    a=np.zeros(shape,np.uint8); b=a.copy()
    a[10:30,10:20]=255; a[30:70,14:16]=255; a[65:67,14:50]=255
    b[10:30,80:90]=255; b[30:70,84:86]=255; b[65:67,52:86]=255
    if case=='fully_hidden': b[:50,:]=0
    if case=='hidden_overlap': b[65:67,48:86]=255
    if case=='visible_overlap': b[10:30,18:22]=255
    if case=='all_visible': win[:,:]=255
    if case=='empty_window': win[:,:]=0
    Image.fromarray(win).save(root/'window.png')
    for name,alpha in [('A',a),('B',b)]:
        rgba=np.zeros((*shape,4),np.uint8);rgba[:,:,:3]=255;rgba[:,:,3]=alpha
        Image.fromarray(rgba).save(root/(name+'.png'))
    return dict(px_per_mm=10,window=str(root/'window.png'),
                items=[f'A={root}/A.png',f'B={root}/B.png'], min_item_gap=6,
                clear_sides=())

def measure(module,args):
    return module.audit(**args)

class VisibleGapTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_hidden_handles_do_not_control_visible_gap(self):
        args=fixture(self.root,'hidden_close');after=measure(new,args)
        self.assertEqual(after['adjacent_visible_gaps_mm'][0]['gap_mm'],6.1)
        self.assertEqual(after['adjacent_full_layer_gaps_mm'][0]['gap_mm'],0.3)
    def test_fully_hidden_is_unknown_not_pass(self):
        args=fixture(self.root,'fully_hidden');after=measure(new,args)
        self.assertIsNone(after['adjacent_visible_gaps_mm'][0]['gap_mm'])
        self.assertEqual(after['summary'],{'ok':0,'warn':1,'fail':0})
        self.assertNotIn('None mm',new.to_markdown(after))
    def test_hidden_overlap(self):
        args=fixture(self.root,'hidden_overlap');after=measure(new,args)
        self.assertEqual(after['adjacent_visible_gaps_mm'][0]['gap_mm'],6.1)
    def test_visible_overlap_still_zero(self):
        args=fixture(self.root,'visible_overlap');after=measure(new,args)
        self.assertEqual(after['adjacent_visible_gaps_mm'][0]['gap_mm'],0)
        self.assertEqual(after['summary']['fail'],1)
    def test_all_visible_unchanged(self):
        args=fixture(self.root,'all_visible')
        self.assertEqual(measure(new,args)['adjacent_visible_gaps_mm'][0]['gap_mm'],0.3)
    def test_empty_window(self):
        args=fixture(self.root,'empty_window');after=measure(new,args)
        self.assertIsNone(after['adjacent_visible_gaps_mm'][0]['gap_mm'])
        self.assertEqual(after['summary']['warn'],1)
    def test_no_window_does_not_claim_visible(self):
        args=fixture(self.root,'hidden_close');args['window']=None
        after=measure(new,args)
        self.assertIsNone(after['adjacent_visible_gaps_mm'][0]['gap_mm'])
        self.assertEqual(after['adjacent_full_layer_gaps_mm'][0]['gap_mm'],0.3)
    def test_thresholds_unchanged(self):
        import inspect
        for name in ('min_window_clear','min_fold_clear','min_ppi','min_item_gap'):
            self.assertEqual({'min_window_clear':5.0,'min_fold_clear':5.0,'min_ppi':300.0,'min_item_gap':None}[name],inspect.signature(new.audit).parameters[name].default)
    def test_no_gap_threshold_no_new_approval(self):
        args=fixture(self.root,'hidden_close');args['min_item_gap']=None
        self.assertEqual(measure(new,args)['summary'],{'ok':0,'warn':0,'fail':0})

if __name__=='__main__':unittest.main()
