from typing import Dict, List, Set
import os
from pathlib import Path
from zipfile import ZipFile
import wget


class StanfordNLP:
    def __init__(self, core_nlp_version: str = '2018-10-05', threads: int = 5, close_after_finish: bool = True):
        self.remote_url = 'http://nlp.stanford.edu/software/stanford-corenlp-full-{}.zip'.format(core_nlp_version)
        self.install_dir = Path(os.environ['STANFORD_HOME']).expanduser()
        self.install_dir.mkdir(exist_ok=True)
        if not (self.install_dir / Path('stanford-corenlp-full-{}'.format(core_nlp_version))).exists():
            print('Downloading to %s.' % self.install_dir)
            output_filename = wget.download(self.remote_url, out=str(self.install_dir))
            print('\nExtracting to %s.' % self.install_dir)
            zf = ZipFile(output_filename)
            zf.extractall(path=self.install_dir)
            zf.close()
        os.environ['CORENLP_HOME'] = str(self.install_dir / 'stanford-corenlp-full-2018-10-05')
        from stanfordnlp.server import CoreNLPClient
        self.close_after_finish = close_after_finish
        self.client = CoreNLPClient(annotators=['parse'], memory='8G', threads=threads)

    def get_parse(self, output):
        for sentence in output['sentences']:
            return sentence['parse']
        return None

    def annotate(self,
                 text: str,
                 properties_key: str = None,
                 properties: dict = None):
        # https://stanfordnlp.github.io/CoreNLP/openie.html
        core_nlp_output = self.client.annotate(text=text, annotators=['parse'], output_format='json',
                                               properties_key=properties_key, properties=properties)
        return self.get_parse(core_nlp_output)
