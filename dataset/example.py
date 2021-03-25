from typing import List, Tuple
import sling
import time
import os
from collections import defaultdict
import numpy as np


class SlingExtractor(object):
  def load_kb(self, root_dir: str='local/data/e/wiki'):
    print('loading and indexing kb ...')
    start = time.time()
    self.kb = sling.Store()
    self.kb.load(os.path.join(root_dir, 'kb.sling'))
    self.phrase = sling.PhraseTable(self.kb, os.path.join(root_dir, 'en', 'phrase-table.repo'))
    self.kb.freeze()
    self.extract_property_names()
    print('loading took', (time.time() - start), 'secs')


  def extract_property_names(self):
    print('storing property names ...')
    start = time.time()
    self.property_names = defaultdict(list)
    for frame in self.kb:
      if 'id' in frame and frame.id.startswith('P'):
        self.property_names[frame.id].append(frame.name)
    print('found', str(len(self.property_names)), 'properties')
    print('took', (time.time() - start), 'sec')


  @staticmethod
  def get_frame_id(frame):
    if 'id' in frame:
      return frame.id
    if 'is' in frame:
      if type(frame['is']) != sling.Frame:
        return None
      if 'id' in frame['is']:
        return frame['is'].id
    return None


  @staticmethod
  def get_date_property(prop, tail):
    if 'target' not in prop:
      return None
    if prop.target.id != '/w/time':
      return None
    prop_id = SlingExtractor.get_frame_id(prop)
    if type(tail) == int:
      return (prop_id, tail)
    elif type(tail) == sling.Frame and 'is' in tail and type(tail['is']) == int:
      return (prop_id, tail['is'])
    return None


  @staticmethod
  def get_canonical_property(prop, tail):
    if type(prop) != sling.Frame or type(tail) != sling.Frame:
      return None
    prop_id = SlingExtractor.get_frame_id(prop)
    tail_id = SlingExtractor.get_frame_id(tail)
    if prop_id is None:
      return None
    if tail_id is None:
      return None
    if not prop_id.startswith('P') or not tail_id.startswith('Q'):
      return None
    return (prop_id, tail_id)


  def get_type(self, wid) -> str:
    for type_prop in ['P31', 'P279']:
      try:
        return self.kb[self.kb[wid][type_prop].id].name
      except:
        pass
    return None


  def get_name(self, wid) -> str:
    return self.kb[wid].name


  def iter_property(self, wid: str, type: str='can', shuffle: bool=False):
    tup_li: List[Tuple] = []
    for prop, tail in self.kb[wid]:
      tup = self.get_canonical_property(prop, tail)
      if tup is not None and type == 'can':
        if not hasattr(self, 'filter') or tup[0] in self.filter:
          tup_li.append(tup)
        continue
      if tup is None:
        tup = self.get_date_property(prop, tail)
        if tup is not None and type == 'time':
          if not hasattr(self, 'filter') or tup[0] in self.filter:
            tup_li.append(tup)
          continue
    group = defaultdict(list)
    for k, v in tup_li:
      group[k].append(v)
    result = list(group.items())
    if shuffle:
      np.random.shuffle(result)
    return result


if __name__ == '__main__':
  se = SlingExtractor()
  se.load_kb(root_dir='/home/zhengbaj/tir4/sling/local/data/e/wiki')
  # the kb is basically a dict with entity id as key and properties ad value
  print(se.kb['Q31'])
  # all the canonical properties
  ps = se.iter_property('Q31', type='can')
