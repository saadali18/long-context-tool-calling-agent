"""Logging tests: privacy, exception tracing, configuration."""
import logging
import tempfile
import unittest
from pathlib import Path
from harness.debug_log import configure_logging, trace

class DebugLoggingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)/"run.log"
        self.log = configure_logging("DEBUG",self.path)
    def tearDown(self):
        configure_logging("WARNING","-")
        self.tmp.cleanup()
    def test_no_arguments_or_results(self):
        @trace
        def echo(secret):
            return secret
        self.assertEqual(echo("SECRET_PAYLOAD"),"SECRET_PAYLOAD")
        contents=self.path.read_text()
        self.assertIn("enter fn=",contents)
        self.assertIn("exit fn=",contents)
        self.assertNotIn("SECRET_PAYLOAD",contents)
    def test_exception_type_not_exception_message(self):
        @trace
        def fail():
            raise ValueError("SENSITIVE_EXCEPTION")
        with self.assertRaises(ValueError):
            fail()
        contents=self.path.read_text()
        self.assertIn("type=ValueError",contents)
        self.assertNotIn("SENSITIVE_EXCEPTION",contents)
    def test_reconfiguration_no_duplicate_handlers(self):
        configure_logging("DEBUG",self.path)
        handlers=[h for h in logging.getLogger("harness").handlers if getattr(h,"_a1_handler",False)]
        self.assertEqual(len(handlers),2)
