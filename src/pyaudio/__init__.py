class PyAudio:
    def open(self, *args, **kwargs):
        class Dummy:
            def read(self, *a, **k):
                return b''
            def stop_stream(self):
                pass
            def close(self):
                pass
        return Dummy()
    def terminate(self):
        pass
