import sys
import unittest
from unittest.mock import patch

from peleng.__main__ import main


class CliTests(unittest.TestCase):
    def test_batch_ui_starts_local_browser_server_without_headless_processing(self):
        created=[]
        class Engine:
            def __init__(self,*args,**kwargs):
                self.options=kwargs
                created.append(self)
            def start(self):raise AssertionError('UI processing must wait for the Start button')
        with patch.object(sys,'argv',['peleng','--batch-ui']), \
                patch('peleng.__main__.Engine',Engine), \
                patch('peleng.inputs.inspect_dataset',return_value={}), \
                patch('uvicorn.run') as serve:
            main()
        self.assertEqual(created[0].options['replay_mode'],'batch')
        self.assertEqual(serve.call_args.kwargs['host'],'127.0.0.1')


if __name__=='__main__':unittest.main()
